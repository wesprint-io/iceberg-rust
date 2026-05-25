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

//! Translation from iceberg's table-level [`SortOrder`] into a DataFusion
//! [`LexOrdering`] suitable for a per-file read scan's
//! `EquivalenceProperties`.

use std::sync::Arc;

use datafusion::arrow::compute::SortOptions;
use datafusion::arrow::datatypes::Schema as ArrowSchema;
use datafusion::common::ScalarValue;
use datafusion::common::config::ConfigOptions;
use datafusion::functions::core::get_field;
use datafusion::physical_expr::expressions::{Column, Literal};
use datafusion::physical_expr::{LexOrdering, PhysicalExpr, PhysicalSortExpr, ScalarFunctionExpr};
use iceberg::spec::{NullOrder, Schema as IcebergSchema, SortDirection, SortOrder, Transform};

/// Translate iceberg's [`SortOrder`] into a DataFusion [`LexOrdering`]
/// over `projected_arrow_schema`.
///
/// Only identity-transformed sort fields produce row-level orderings:
/// non-identity transforms (`days`, `bucket`, etc.) constrain
/// partitioning, not within-file row order. On the first non-identity
/// transform the translation stops and the prefix already collected
/// is returned (which is still a valid sub-ordering).
///
/// Nested-leaf sort fields (e.g. `payload.user_uid`) are emitted as a
/// flattened `get_field(base, "part1", "part2", ...)`
/// [`ScalarFunctionExpr`] — the shape DataFusion's SQL planner
/// produces for `ORDER BY base.part1.part2` after the
/// `SimplifyExpressions` optimizer pass collapses nested `get_field`
/// calls. Matching that shape is what lets
/// `EnforceSorting` recognize the declared ordering when comparing it
/// against the planner's required ordering.
///
/// Returns `None` if no usable prefix exists — unsorted order, a
/// non-identity primary sort field, or a projected schema that
/// dropped the primary sort column.
pub(crate) fn translate_sort_order(
    sort_order: &SortOrder,
    iceberg_schema: &IcebergSchema,
    projected_arrow_schema: &ArrowSchema,
    config_options: Arc<ConfigOptions>,
) -> Option<LexOrdering> {
    if sort_order.is_unsorted() {
        return None;
    }
    let mut exprs = Vec::with_capacity(sort_order.fields.len());
    for field in &sort_order.fields {
        if !matches!(field.transform, Transform::Identity) {
            break;
        }
        let Some(expr) = build_sort_expr(
            field.source_id,
            iceberg_schema,
            projected_arrow_schema,
            &config_options,
        ) else {
            break;
        };
        let options = SortOptions {
            descending: matches!(field.direction, SortDirection::Descending),
            nulls_first: matches!(field.null_order, NullOrder::First),
        };
        exprs.push(PhysicalSortExpr { expr, options });
    }
    LexOrdering::new(exprs)
}

/// Build the [`PhysicalExpr`] for one sort field's `source_id`, mapped
/// against `projected`. Top-level identity fields become a [`Column`].
/// Nested-leaf fields become a flattened `get_field` call.
fn build_sort_expr(
    source_id: i32,
    iceberg_schema: &IcebergSchema,
    projected: &ArrowSchema,
    config_options: &Arc<ConfigOptions>,
) -> Option<Arc<dyn PhysicalExpr>> {
    let dotted = iceberg_schema.name_by_field_id(source_id)?;
    let mut parts = dotted.split('.');
    let head = parts.next()?;
    let (idx, _) = projected.column_with_name(head)?;
    let root: Arc<dyn PhysicalExpr> = Arc::new(Column::new(head, idx));

    let rest: Vec<&str> = parts.collect();
    if rest.is_empty() {
        return Some(root);
    }

    // Flattened `get_field(base, "p1", "p2", ...)`: matches the shape
    // SimplifyExpressions produces from nested SQL access like `a.b.c`.
    let mut args: Vec<Arc<dyn PhysicalExpr>> = Vec::with_capacity(rest.len() + 1);
    args.push(root);
    for part in rest {
        args.push(Arc::new(Literal::new(ScalarValue::Utf8(Some(
            part.to_string(),
        )))));
    }
    let udf = get_field();
    let expr =
        ScalarFunctionExpr::try_new(udf, args, projected, Arc::clone(config_options)).ok()?;
    Some(Arc::new(expr))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::datatypes::{DataType, Field, Fields, Schema as ArrowSchema};
    use datafusion::common::config::ConfigOptions;
    use iceberg::spec::{
        NestedField, NullOrder, PrimitiveType, Schema as IcebergSchema, SortDirection, SortField,
        SortOrder, Transform, Type,
    };

    use super::*;

    fn config_options() -> Arc<ConfigOptions> {
        Arc::new(ConfigOptions::default())
    }

    #[test]
    fn unsorted_order_yields_none() {
        let schema = IcebergSchema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .unwrap();
        let arrow = ArrowSchema::new(vec![Field::new("id", DataType::Int64, false)]);
        assert!(
            translate_sort_order(
                &SortOrder::unsorted_order(),
                &schema,
                &arrow,
                config_options()
            )
            .is_none()
        );
    }

    #[test]
    fn top_level_identity_sort_produces_column_ordering() {
        let schema = IcebergSchema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(2, "ts", Type::Primitive(PrimitiveType::TimestamptzNs))
                    .into(),
            ])
            .build()
            .unwrap();
        let arrow = ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new(
                "ts",
                DataType::Timestamp(datafusion::arrow::datatypes::TimeUnit::Nanosecond, None),
                false,
            ),
        ]);

        let order = SortOrder::builder()
            .with_order_id(1)
            .with_sort_field(SortField {
                source_id: 1,
                transform: Transform::Identity,
                direction: SortDirection::Ascending,
                null_order: NullOrder::Last,
            })
            .with_sort_field(SortField {
                source_id: 2,
                transform: Transform::Identity,
                direction: SortDirection::Descending,
                null_order: NullOrder::First,
            })
            .build(&schema)
            .unwrap();

        let lex = translate_sort_order(&order, &schema, &arrow, config_options()).unwrap();
        assert_eq!(lex.len(), 2);

        let first = &lex[0];
        let col = first.expr.as_any().downcast_ref::<Column>().unwrap();
        assert_eq!(col.name(), "id");
        assert_eq!(col.index(), 0);
        assert!(!first.options.descending);
        assert!(!first.options.nulls_first);

        let second = &lex[1];
        let col = second.expr.as_any().downcast_ref::<Column>().unwrap();
        assert_eq!(col.name(), "ts");
        assert!(second.options.descending);
        assert!(second.options.nulls_first);
    }

    #[test]
    fn non_identity_transform_truncates_prefix() {
        let schema = IcebergSchema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(2, "ts", Type::Primitive(PrimitiveType::TimestamptzNs))
                    .into(),
            ])
            .build()
            .unwrap();
        let arrow = ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new(
                "ts",
                DataType::Timestamp(datafusion::arrow::datatypes::TimeUnit::Nanosecond, None),
                false,
            ),
        ]);

        // `days(ts)` is a partition transform, not an in-file ordering.
        // It appears AFTER an identity prefix on `id`, so the translator
        // should yield just the `id` ordering.
        let order = SortOrder::builder()
            .with_order_id(1)
            .with_sort_field(SortField {
                source_id: 1,
                transform: Transform::Identity,
                direction: SortDirection::Ascending,
                null_order: NullOrder::Last,
            })
            .with_sort_field(SortField {
                source_id: 2,
                transform: Transform::Day,
                direction: SortDirection::Ascending,
                null_order: NullOrder::Last,
            })
            .build(&schema)
            .unwrap();

        let lex = translate_sort_order(&order, &schema, &arrow, config_options()).unwrap();
        assert_eq!(lex.len(), 1, "only the identity prefix survives");
    }

    #[test]
    fn nested_leaf_sort_produces_get_field_expr() {
        // Build a schema with `payload: struct { user_uid: long }`. The
        // sort key references the nested leaf by its field id.
        let payload = NestedField::required(
            1,
            "payload",
            Type::Struct(iceberg::spec::StructType::new(vec![Arc::new(
                NestedField::required(2, "user_uid", Type::Primitive(PrimitiveType::Long)),
            )])),
        );
        let schema = IcebergSchema::builder()
            .with_schema_id(0)
            .with_fields(vec![Arc::new(payload)])
            .build()
            .unwrap();
        let arrow = ArrowSchema::new(vec![Field::new(
            "payload",
            DataType::Struct(Fields::from(vec![Field::new(
                "user_uid",
                DataType::Int64,
                false,
            )])),
            false,
        )]);

        let order = SortOrder::builder()
            .with_order_id(1)
            .with_sort_field(SortField {
                source_id: 2,
                transform: Transform::Identity,
                direction: SortDirection::Ascending,
                null_order: NullOrder::Last,
            })
            .build(&schema)
            .unwrap();

        let lex = translate_sort_order(&order, &schema, &arrow, config_options()).unwrap();
        assert_eq!(lex.len(), 1);

        let sf = lex[0]
            .expr
            .as_any()
            .downcast_ref::<ScalarFunctionExpr>()
            .expect("nested access must be a ScalarFunctionExpr");
        assert_eq!(sf.fun().name(), "get_field");

        let args = sf.args();
        assert_eq!(args.len(), 2);
        let base = args[0].as_any().downcast_ref::<Column>().unwrap();
        assert_eq!(base.name(), "payload");
        let path = args[1].as_any().downcast_ref::<Literal>().unwrap();
        assert_eq!(
            path.value(),
            &ScalarValue::Utf8(Some("user_uid".to_string()))
        );
    }
}
