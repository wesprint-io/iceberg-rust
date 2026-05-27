// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Physical optimizer rules for the iceberg-datafusion integration.
//!
//! ## [`NestedFieldProjectionPushdown`]
//!
//! DataFusion's [`TableProvider::scan`] API conveys projections as top-level
//! column indices (`Option<&Vec<usize>>`), so a query like
//! `SELECT outer.inner.leaf FROM tbl` lowers to a `ProjectionExec` that
//! extracts the leaf via `get_field` from the *full* top-level struct produced
//! by the [`IcebergTableScan`]. iceberg-rust's `TableScan::select` already
//! accepts dotted nested paths and narrows the parquet projection mask to
//! exactly those leaves, but the integration has no channel to communicate
//! the nested path through `TableProvider::scan`.
//!
//! This rule closes the gap. After physical planning, it walks the plan tree
//! looking for `ProjectionExec`s whose every output expression is a chain of
//! `get_field` calls rooted at a column produced by an [`IcebergTableScan`]
//! immediately below (allowing one `CooperativeExec` wrapper between them).
//! When the pattern matches, the rule rebuilds the scan via
//! [`IcebergTableScan::with_nested_projection`] using the dotted paths
//! harvested from the `get_field` chains, declaring a narrowed arrow output
//! schema that matches what the iceberg arrow reader produces for those
//! paths.
//!
//! [`TableProvider::scan`]: datafusion::datasource::TableProvider::scan
//! [`IcebergTableScan`]: crate::physical_plan::IcebergTableScan
//! [`IcebergTableScan::with_nested_projection`]: crate::physical_plan::IcebergTableScan::with_nested_projection

use std::collections::BTreeMap;
use std::sync::Arc;

use datafusion::arrow::datatypes::{
    DataType, Field, FieldRef, Fields, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef,
};
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion::error::Result as DFResult;
use datafusion::physical_expr::ScalarFunctionExpr;
use datafusion::physical_expr::expressions::{Column, Literal};
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::coop::CooperativeExec;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::{ExecutionPlan, PhysicalExpr};
use datafusion::scalar::ScalarValue;

use crate::physical_plan::IcebergTableScan;

/// Pushes nested-field projections (`get_field` chains) into the underlying
/// [`IcebergTableScan`], so iceberg reads only the requested nested leaves
/// from each Parquet file instead of the entire top-level struct.
///
/// Register on a `SessionStateBuilder` via
/// `with_physical_optimizer_rule(Arc::new(NestedFieldProjectionPushdown))`.
#[derive(Default, Debug)]
pub struct NestedFieldProjectionPushdown;

impl PhysicalOptimizerRule for NestedFieldProjectionPushdown {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        plan.transform_up(try_rewrite).data()
    }

    fn name(&self) -> &str {
        "iceberg_nested_field_projection_pushdown"
    }

    fn schema_check(&self) -> bool {
        // The rewrite narrows the scan's declared struct types. Top-level
        // column count and the projection's output types are preserved
        // (`get_field` chains return the same leaf types as before), but the
        // intermediate scan schema differs, so the global schema-equality
        // check would falsely flag this rule.
        false
    }
}

fn try_rewrite(plan: Arc<dyn ExecutionPlan>) -> DFResult<Transformed<Arc<dyn ExecutionPlan>>> {
    let Some(projection) = plan.as_any().downcast_ref::<ProjectionExec>() else {
        return Ok(Transformed::no(plan));
    };

    let Some(scan_site) = find_iceberg_scan(projection.input()) else {
        return Ok(Transformed::no(plan));
    };

    // Extract dotted paths from every projection expression. If any expression
    // is not a `get_field` chain rooted at a Column from the scan's output,
    // bail out (a bare Column would require leaving its top-level field
    // unnarrowed, which the MVP doesn't model).
    let scan_schema = scan_site.scan.schema();
    let mut paths_per_column: BTreeMap<usize, Vec<Vec<String>>> = BTreeMap::new();

    for proj_expr in projection.expr() {
        let Some((column, path)) = extract_get_field_chain(&proj_expr.expr) else {
            return Ok(Transformed::no(plan));
        };
        if path.is_empty() {
            // Bare column reference — would have to keep the entire top-level
            // column. Not handled in this rule.
            return Ok(Transformed::no(plan));
        }
        // Verify the column actually comes from the scan we're rewriting.
        if column.index() >= scan_schema.fields().len() {
            return Ok(Transformed::no(plan));
        }
        if scan_schema.field(column.index()).name() != column.name() {
            return Ok(Transformed::no(plan));
        }
        paths_per_column
            .entry(column.index())
            .or_default()
            .push(path);
    }

    if paths_per_column.is_empty() {
        return Ok(Transformed::no(plan));
    }

    // Build the flat list of dotted paths and the narrowed arrow schema.
    let mut dotted_paths: Vec<String> = Vec::new();
    let mut narrowed_fields: Vec<FieldRef> = Vec::new();
    for (col_idx, paths) in &paths_per_column {
        let field = scan_schema.field(*col_idx);
        for path in paths {
            dotted_paths.push(prefix_with_field_name(field.name(), path));
        }
        let narrowed = match narrow_field(field, paths) {
            Some(f) => f,
            None => return Ok(Transformed::no(plan)),
        };
        narrowed_fields.push(Arc::new(narrowed));
    }
    let narrowed_schema: ArrowSchemaRef = Arc::new(ArrowSchema::new_with_metadata(
        narrowed_fields,
        scan_schema.metadata().clone(),
    ));

    // Build the rewritten scan and re-attach any wrapper (e.g. CooperativeExec)
    // that sat between the projection and the scan.
    let new_scan = scan_site
        .scan
        .with_nested_projection(dotted_paths, narrowed_schema)?;
    let new_input: Arc<dyn ExecutionPlan> = scan_site.rewrap(Arc::new(new_scan));

    // Reuse the original projection expressions: their Column refs still point
    // at the same top-level field name/index, and their cached `get_field`
    // return types match the leaves (which are preserved verbatim under the
    // narrowed struct). ProjectionExec recomputes its own output schema from
    // the new input.
    let proj_exprs: Vec<(Arc<dyn PhysicalExpr>, String)> = projection
        .expr()
        .iter()
        .map(|e| (e.expr.clone(), e.alias.clone()))
        .collect();
    let new_projection = ProjectionExec::try_new(proj_exprs, new_input)?;
    Ok(Transformed::yes(Arc::new(new_projection)))
}

/// One [`IcebergTableScan`] plus any single wrapper (e.g. [`CooperativeExec`])
/// we walked through to find it. `rewrap` puts the wrapper back over a
/// rewritten scan.
struct ScanSite<'a> {
    scan: &'a IcebergTableScan,
    wrapper: ScanWrapper,
}

enum ScanWrapper {
    None,
    Cooperative,
}

impl<'a> ScanSite<'a> {
    fn rewrap(&self, scan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        match self.wrapper {
            ScanWrapper::None => scan,
            ScanWrapper::Cooperative => Arc::new(CooperativeExec::new(scan)),
        }
    }
}

fn find_iceberg_scan(node: &Arc<dyn ExecutionPlan>) -> Option<ScanSite<'_>> {
    if let Some(scan) = node.as_any().downcast_ref::<IcebergTableScan>() {
        return Some(ScanSite {
            scan,
            wrapper: ScanWrapper::None,
        });
    }
    if node.as_any().downcast_ref::<CooperativeExec>().is_some() {
        let inner = node.children().into_iter().next()?;
        if let Some(scan) = inner.as_any().downcast_ref::<IcebergTableScan>() {
            return Some(ScanSite {
                scan,
                wrapper: ScanWrapper::Cooperative,
            });
        }
    }
    None
}

/// Walks a `get_field(...)` chain rooted at a Column and returns
/// `(column, [field1, field2, ...])`. The path is empty when `expr` is just a
/// bare column reference; `None` when the expression isn't a column-rooted
/// chain of `get_field` calls with literal string field-name arguments.
///
/// Handles both shapes datafusion can produce:
///   - nested two-arg calls: `get_field(get_field(c, 'a'), 'b')`
///   - one variadic call:    `get_field(c, 'a', 'b')`
fn extract_get_field_chain(expr: &Arc<dyn PhysicalExpr>) -> Option<(Column, Vec<String>)> {
    if let Some(col) = expr.as_any().downcast_ref::<Column>() {
        return Some((col.clone(), Vec::new()));
    }
    let scalar = expr.as_any().downcast_ref::<ScalarFunctionExpr>()?;
    if scalar.name() != "get_field" || scalar.args().len() < 2 {
        return None;
    }
    let args = scalar.args();
    let (column, mut path) = extract_get_field_chain(&args[0])?;
    for arg in &args[1..] {
        let lit = arg.as_any().downcast_ref::<Literal>()?;
        let field_name = match lit.value() {
            ScalarValue::Utf8(Some(name))
            | ScalarValue::LargeUtf8(Some(name))
            | ScalarValue::Utf8View(Some(name)) => name.clone(),
            _ => return None,
        };
        path.push(field_name);
    }
    Some((column, path))
}

fn prefix_with_field_name(top_level: &str, rest: &[String]) -> String {
    let mut out =
        String::with_capacity(top_level.len() + rest.iter().map(|s| s.len() + 1).sum::<usize>());
    out.push_str(top_level);
    for segment in rest {
        out.push('.');
        out.push_str(segment);
    }
    out
}

/// Build a narrowed arrow `Field` that retains only the nested children
/// reachable from `paths` (paths are relative to `field`). Returns `None` if
/// any path references a child that doesn't exist on `field`'s struct.
fn narrow_field(field: &Field, paths: &[Vec<String>]) -> Option<Field> {
    // Any path equal to the empty vec means "keep this whole subtree" — but
    // the caller already rejects empty paths at the top, and recursion only
    // creates empty paths when a request bottoms out exactly at a struct.
    if paths.iter().any(|p| p.is_empty()) {
        return Some(field.clone());
    }
    match field.data_type() {
        DataType::Struct(children) => {
            let kept = narrow_struct_children(children, paths)?;
            Some(
                Field::new(field.name(), DataType::Struct(kept), field.is_nullable())
                    .with_metadata(field.metadata().clone()),
            )
        }
        // Anything else (primitive, list, map) can't have its interior pruned
        // by a dotted-path projection; the caller shouldn't have asked for a
        // sub-path of it. Reject so the rule bails out safely.
        _ => None,
    }
}

fn narrow_struct_children(children: &Fields, paths: &[Vec<String>]) -> Option<Fields> {
    let mut kept: Vec<FieldRef> = Vec::new();
    for child in children.iter() {
        let child_paths: Vec<Vec<String>> = paths
            .iter()
            .filter(|p| !p.is_empty() && p[0] == *child.name())
            .map(|p| p[1..].to_vec())
            .collect();
        if child_paths.is_empty() {
            continue;
        }
        if child_paths.iter().any(|p| p.is_empty()) {
            // Some request keeps this child entirely.
            kept.push(child.clone());
        } else {
            let narrowed = narrow_field(child, &child_paths)?;
            kept.push(Arc::new(narrowed));
        }
    }
    if kept.is_empty() {
        None
    } else {
        Some(kept.into())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use datafusion::arrow::datatypes::{DataType, Field, Fields};
    use datafusion::common::ScalarValue;
    use datafusion::functions::core::expr_fn::get_field_path;
    use datafusion::logical_expr::{Expr, col, lit};
    use datafusion::physical_expr::expressions::{Column, Literal};
    use datafusion::physical_expr::planner::create_physical_expr;
    use datafusion::physical_expr::{PhysicalExpr, ScalarFunctionExpr};
    use datafusion::physical_plan::ExecutionPlan;
    use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
    use datafusion::physical_plan::empty::EmptyExec;
    use datafusion::physical_plan::projection::ProjectionExec;
    use datafusion::prelude::SessionContext;
    use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

    use super::*;

    fn field_with_id(name: &str, data_type: DataType, id: u32) -> Field {
        Field::new(name, data_type, true).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            id.to_string(),
        )]))
    }

    /// Schema with a nested struct:
    /// ```text
    /// outer: struct {
    ///   a: int,
    ///   b: struct {
    ///     c: bigint,
    ///     d: string,
    ///   },
    /// }
    /// ```
    fn nested_arrow_schema() -> ArrowSchemaRef {
        let inner = Fields::from(vec![
            field_with_id("c", DataType::Int64, 7),
            field_with_id("d", DataType::Utf8, 8),
        ]);
        let outer_children = Fields::from(vec![
            field_with_id("a", DataType::Int32, 5),
            field_with_id("b", DataType::Struct(inner), 6),
        ]);
        Arc::new(ArrowSchema::new(vec![field_with_id(
            "outer",
            DataType::Struct(outer_children),
            10,
        )]))
    }

    #[test]
    fn narrow_field_keeps_only_requested_paths() {
        let schema = nested_arrow_schema();
        let outer = schema.field(0);
        let paths = vec![vec!["b".to_string(), "c".to_string()]];
        let narrowed = narrow_field(outer, &paths).unwrap();
        let DataType::Struct(top_children) = narrowed.data_type() else {
            panic!("outer should still be a struct");
        };
        assert_eq!(top_children.len(), 1, "only `b` should survive");
        let b = &top_children[0];
        assert_eq!(b.name(), "b");
        let DataType::Struct(b_children) = b.data_type() else {
            panic!("b should still be a struct");
        };
        assert_eq!(b_children.len(), 1, "only `b.c` should survive under `b`");
        assert_eq!(b_children[0].name(), "c");
    }

    #[test]
    fn narrow_field_unions_sibling_paths() {
        let schema = nested_arrow_schema();
        let outer = schema.field(0);
        let paths = vec![vec!["b".to_string(), "c".to_string()], vec![
            "b".to_string(),
            "d".to_string(),
        ]];
        let narrowed = narrow_field(outer, &paths).unwrap();
        let DataType::Struct(top_children) = narrowed.data_type() else {
            panic!();
        };
        assert_eq!(top_children.len(), 1);
        let DataType::Struct(b_children) = top_children[0].data_type() else {
            panic!();
        };
        let names: Vec<&str> = b_children.iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, vec!["c", "d"]);
    }

    /// Build a physical `get_field` chain over a column at the given index, so
    /// we can exercise the chain walker without a full SQL pipeline.
    fn build_get_field_chain(
        input_schema: &ArrowSchemaRef,
        col_name: &str,
        path: &[&str],
    ) -> Arc<dyn PhysicalExpr> {
        let logical: Expr = path.iter().skip(1).fold(
            get_field_path(col(col_name), vec![lit(path[0])]),
            |acc, segment| get_field_path(acc, vec![lit(*segment)]),
        );
        let df_schema =
            datafusion::common::DFSchema::try_from(ArrowSchema::clone(input_schema)).unwrap();
        let ctx = SessionContext::new();
        create_physical_expr(&logical, &df_schema, ctx.state().execution_props()).unwrap()
    }

    #[test]
    fn extract_chain_recovers_path_from_nested_get_field_calls() {
        let schema = nested_arrow_schema();
        let expr = build_get_field_chain(&schema, "outer", &["b", "c"]);
        let (column, path) = extract_get_field_chain(&expr).expect("chain should be recognized");
        assert_eq!(column.name(), "outer");
        assert_eq!(column.index(), 0);
        assert_eq!(path, vec!["b".to_string(), "c".to_string()]);
    }

    #[test]
    fn extract_chain_handles_variadic_form() {
        // Construct a variadic `get_field(outer, "b", "c")` physical expression
        // directly, since the logical builder may flatten or not depending on
        // datafusion version. This ensures the walker handles both shapes.
        let column: Arc<dyn PhysicalExpr> = Arc::new(Column::new("outer", 0));
        let args: Vec<Arc<dyn PhysicalExpr>> = vec![
            column,
            Arc::new(Literal::new(ScalarValue::Utf8(Some("b".to_string())))),
            Arc::new(Literal::new(ScalarValue::Utf8(Some("c".to_string())))),
        ];
        let get_field_udf = datafusion::functions::core::get_field();
        let return_type = DataType::Int64;
        let scalar = ScalarFunctionExpr::new(
            "get_field",
            get_field_udf,
            args,
            Arc::new(Field::new("ignored", return_type, true)),
            Arc::new(ConfigOptions::new()),
        );
        let expr: Arc<dyn PhysicalExpr> = Arc::new(scalar);
        let (col, path) = extract_get_field_chain(&expr).expect("variadic chain recognized");
        assert_eq!(col.name(), "outer");
        assert_eq!(path, vec!["b".to_string(), "c".to_string()]);
    }

    #[test]
    fn extract_chain_rejects_bare_column() {
        let column: Arc<dyn PhysicalExpr> = Arc::new(Column::new("outer", 0));
        let (col, path) = extract_get_field_chain(&column).unwrap();
        assert_eq!(col.name(), "outer");
        assert!(path.is_empty());
    }

    #[test]
    fn rule_leaves_unrelated_plans_alone() {
        // ProjectionExec over an EmptyExec (no IcebergTableScan) must be
        // returned unchanged by the rule.
        let schema = nested_arrow_schema();
        let input: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(schema.clone()));
        let proj_expr: Arc<dyn PhysicalExpr> = Arc::new(Column::new("outer", 0));
        let projection = Arc::new(
            ProjectionExec::try_new(vec![(proj_expr, "outer".to_string())], input).unwrap(),
        );
        let rule = NestedFieldProjectionPushdown;
        let optimized = rule
            .optimize(projection.clone(), &ConfigOptions::new())
            .unwrap();
        assert!(Arc::ptr_eq(
            &(projection as Arc<dyn ExecutionPlan>),
            &optimized
        ));
    }

    #[test]
    fn coalesce_partitions_is_not_an_iceberg_scan_wrapper() {
        let input: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(nested_arrow_schema()));
        let wrapped: Arc<dyn ExecutionPlan> = Arc::new(CoalescePartitionsExec::new(input));
        assert!(find_iceberg_scan(&wrapped).is_none());
    }

    // ---- End-to-end rewrites against a real (empty) iceberg table ----

    use std::collections::HashMap as StdHashMap;

    use datafusion::physical_plan::coop::CooperativeExec;
    use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
    use iceberg::spec::{NestedField, PrimitiveType, Schema as IcebergSchema, StructType, Type};
    use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};
    use tempfile::TempDir;

    use crate::physical_plan::IcebergTableScan;

    async fn empty_nested_iceberg_scan() -> (IcebergTableScan, TempDir) {
        // outer: struct<a: int, b: struct<c: bigint, d: string>>
        let inner = StructType::new(vec![
            NestedField::optional(7, "c", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(8, "d", Type::Primitive(PrimitiveType::String)).into(),
        ]);
        let outer_struct = StructType::new(vec![
            NestedField::optional(5, "a", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(6, "b", Type::Struct(inner)).into(),
        ]);
        let iceberg_schema = IcebergSchema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::optional(10, "outer", Type::Struct(outer_struct)).into(),
            ])
            .build()
            .unwrap();

        let temp_dir = TempDir::new().unwrap();
        let warehouse = temp_dir.path().to_str().unwrap().to_string();
        let catalog = MemoryCatalogBuilder::default()
            .load(
                "memory",
                StdHashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
            )
            .await
            .unwrap();
        let namespace = NamespaceIdent::new("ns".to_string());
        catalog
            .create_namespace(&namespace, StdHashMap::new())
            .await
            .unwrap();
        let creation = TableCreation::builder()
            .name("t".to_string())
            .location(format!("{warehouse}/t"))
            .schema(iceberg_schema)
            .properties(StdHashMap::new())
            .build();
        catalog.create_table(&namespace, creation).await.unwrap();

        let table = catalog
            .load_table(&iceberg::TableIdent::new(namespace, "t".to_string()))
            .await
            .unwrap();
        let arrow_schema = nested_arrow_schema();
        let scan = IcebergTableScan::try_new(table, None, arrow_schema, None, &[], None)
            .await
            .unwrap();
        (scan, temp_dir)
    }

    #[tokio::test]
    async fn rule_rewrites_scan_to_dotted_path() {
        let (scan, _td) = empty_nested_iceberg_scan().await;
        let scan_schema = scan.schema();

        // ProjectionExec: [get_field(outer, "b", "c") AS leaf]
        let leaf = build_get_field_chain(&scan_schema, "outer", &["b", "c"]);
        let projection: Arc<dyn ExecutionPlan> = Arc::new(
            ProjectionExec::try_new(
                vec![(leaf, "leaf".to_string())],
                Arc::new(scan) as Arc<dyn ExecutionPlan>,
            )
            .unwrap(),
        );

        let optimized = NestedFieldProjectionPushdown
            .optimize(projection, &ConfigOptions::new())
            .unwrap();

        // The new plan should still be a ProjectionExec; its child must be the
        // rewritten IcebergTableScan with `outer.b.c` in its projection().
        let new_proj = optimized
            .as_any()
            .downcast_ref::<ProjectionExec>()
            .expect("top of rewritten plan is still ProjectionExec");
        let new_scan = new_proj
            .input()
            .as_any()
            .downcast_ref::<IcebergTableScan>()
            .expect("rewritten child is IcebergTableScan");
        let paths = new_scan.projection().expect("projection set after rewrite");
        assert_eq!(paths, ["outer.b.c"], "scan should select only the leaf");
    }

    #[tokio::test]
    async fn rule_walks_through_cooperative_wrapper() {
        let (scan, _td) = empty_nested_iceberg_scan().await;
        let scan_schema = scan.schema();
        let scan_plan: Arc<dyn ExecutionPlan> = Arc::new(scan);
        let coop: Arc<dyn ExecutionPlan> = Arc::new(CooperativeExec::new(scan_plan));

        let leaf = build_get_field_chain(&scan_schema, "outer", &["b", "d"]);
        let projection: Arc<dyn ExecutionPlan> =
            Arc::new(ProjectionExec::try_new(vec![(leaf, "leaf".to_string())], coop).unwrap());

        let optimized = NestedFieldProjectionPushdown
            .optimize(projection, &ConfigOptions::new())
            .unwrap();

        let new_proj = optimized
            .as_any()
            .downcast_ref::<ProjectionExec>()
            .expect("ProjectionExec preserved at top");
        let inner = new_proj
            .input()
            .as_any()
            .downcast_ref::<CooperativeExec>()
            .expect("CooperativeExec wrapper re-attached");
        let new_scan = inner
            .children()
            .into_iter()
            .next()
            .unwrap()
            .as_any()
            .downcast_ref::<IcebergTableScan>()
            .expect("scan beneath the cooperative wrapper");
        let paths = new_scan.projection().unwrap();
        assert_eq!(paths, ["outer.b.d"]);
    }

    #[tokio::test]
    async fn rule_skips_when_any_expression_is_bare_column() {
        let (scan, _td) = empty_nested_iceberg_scan().await;
        let scan_schema = scan.schema();
        let leaf = build_get_field_chain(&scan_schema, "outer", &["b", "c"]);
        let bare: Arc<dyn PhysicalExpr> = Arc::new(Column::new("outer", 0));
        let scan_arc: Arc<dyn ExecutionPlan> = Arc::new(scan);
        let projection: Arc<dyn ExecutionPlan> = Arc::new(
            ProjectionExec::try_new(
                vec![(leaf, "leaf".to_string()), (bare, "outer".to_string())],
                scan_arc.clone(),
            )
            .unwrap(),
        );

        let optimized = NestedFieldProjectionPushdown
            .optimize(projection, &ConfigOptions::new())
            .unwrap();
        let new_proj = optimized
            .as_any()
            .downcast_ref::<ProjectionExec>()
            .expect("still a projection");
        let new_scan = new_proj
            .input()
            .as_any()
            .downcast_ref::<IcebergTableScan>()
            .expect("child still the scan");
        // No narrowing happened — projection stays None (all top-level columns).
        assert!(
            new_scan.projection().is_none(),
            "scan should not have been narrowed when a bare column is present"
        );
    }

    #[tokio::test]
    async fn rule_combines_multiple_paths_from_same_column() {
        let (scan, _td) = empty_nested_iceberg_scan().await;
        let scan_schema = scan.schema();
        let c = build_get_field_chain(&scan_schema, "outer", &["b", "c"]);
        let d = build_get_field_chain(&scan_schema, "outer", &["b", "d"]);
        let projection: Arc<dyn ExecutionPlan> = Arc::new(
            ProjectionExec::try_new(
                vec![(c, "c".to_string()), (d, "d".to_string())],
                Arc::new(scan) as Arc<dyn ExecutionPlan>,
            )
            .unwrap(),
        );

        let optimized = NestedFieldProjectionPushdown
            .optimize(projection, &ConfigOptions::new())
            .unwrap();
        let new_scan = optimized
            .as_any()
            .downcast_ref::<ProjectionExec>()
            .unwrap()
            .input()
            .as_any()
            .downcast_ref::<IcebergTableScan>()
            .unwrap();
        let paths = new_scan.projection().unwrap();
        assert_eq!(paths, ["outer.b.c", "outer.b.d"]);
    }
}
