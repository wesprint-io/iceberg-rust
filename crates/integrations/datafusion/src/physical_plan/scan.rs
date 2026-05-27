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

use std::any::Any;
use std::sync::Arc;
use std::vec;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef as ArrowSchemaRef;
use datafusion::error::Result as DFResult;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, ExecutionPlan, Partitioning, PlanProperties};
use datafusion::prelude::Expr;
use futures::{Stream, TryStreamExt};
use iceberg::arrow::ArrowReaderBuilder;
use iceberg::expr::Predicate;
use iceberg::scan::FileScanTask;
use iceberg::table::Table;

use super::expr_to_predicate::convert_filters_to_predicate;
use crate::to_datafusion_error;

/// Manages the scanning process of an Iceberg [`Table`], encapsulating the
/// necessary details and computed properties required for execution planning.
///
/// Each [`FileScanTask`] is exposed as its own DataFusion partition. This
/// preserves the per-file intrinsic ordering that conforming writers produce
/// (per the table's `write.sort.order`), letting downstream planner rules
/// like `EnforceSorting` insert `SortPreservingMergeExec` instead of forcing
/// a global sort.
#[derive(Debug)]
pub struct IcebergTableScan {
    /// A table in the catalog.
    table: Table,
    /// Snapshot of the table to scan.
    snapshot_id: Option<i64>,
    /// Stores certain, often expensive to compute,
    /// plan properties used in query optimization.
    plan_properties: Arc<PlanProperties>,
    /// Projection column names, None means all columns
    projection: Option<Vec<String>>,
    /// Filters to apply to the table scan
    predicates: Option<Predicate>,
    /// Optional limit on the number of rows to return. Today this is honored
    /// only when there is a single output partition; with multiple partitions
    /// we leave the limit to a wrapping DataFusion `LimitExec`, which is
    /// correct across the merged output.
    limit: Option<usize>,
    /// One [`FileScanTask`] per output partition. Computed eagerly in
    /// [`Self::try_new`] so partition count and ordering can be declared
    /// on `plan_properties`.
    file_scan_tasks: Arc<[FileScanTask]>,
}

impl IcebergTableScan {
    /// Creates a new [`IcebergTableScan`] object by eagerly planning the
    /// underlying file scan against `table`.
    ///
    /// The set of [`FileScanTask`]s is fixed at construction time so the
    /// resulting plan can declare its partition count (one DataFusion
    /// partition per file) to downstream optimizer rules.
    pub(crate) async fn try_new(
        table: Table,
        snapshot_id: Option<i64>,
        schema: ArrowSchemaRef,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Self> {
        let output_schema = match projection {
            None => schema.clone(),
            Some(projection) => Arc::new(schema.project(projection).unwrap()),
        };
        let column_names = get_column_names(schema.clone(), projection);
        let predicates = convert_filters_to_predicate(filters);

        let file_scan_tasks = plan_files(
            &table,
            snapshot_id,
            column_names.clone(),
            predicates.clone(),
        )
        .await?;

        let plan_properties = Self::compute_properties(output_schema, file_scan_tasks.len());

        Ok(Self {
            table,
            snapshot_id,
            plan_properties,
            projection: column_names,
            predicates,
            limit,
            file_scan_tasks,
        })
    }

    pub fn table(&self) -> &Table {
        &self.table
    }

    pub fn snapshot_id(&self) -> Option<i64> {
        self.snapshot_id
    }

    pub fn projection(&self) -> Option<&[String]> {
        self.projection.as_deref()
    }

    pub fn predicates(&self) -> Option<&Predicate> {
        self.predicates.as_ref()
    }

    pub fn limit(&self) -> Option<usize> {
        self.limit
    }

    pub fn file_scan_tasks(&self) -> &[FileScanTask] {
        &self.file_scan_tasks
    }

    /// Returns a scan that reads only the given dotted nested paths from each
    /// file. The current `FileScanTask`s are reused (their data-file set and
    /// predicate are unchanged), but each task's `project_field_ids` is
    /// replaced with the field ids resolved from `dotted_paths` against the
    /// table's current schema. `narrowed_schema` becomes the declared output
    /// schema of the rewritten scan and must match the arrow shape the iceberg
    /// reader will produce for the requested paths (the enclosing struct path
    /// is preserved; only the projected leaves remain inside).
    ///
    /// Used by [`crate::physical_optimizer::NestedFieldProjectionPushdown`] to
    /// recover the nested projection that `TableProvider::scan` cannot carry.
    pub(crate) fn with_nested_projection(
        &self,
        dotted_paths: Vec<String>,
        narrowed_schema: ArrowSchemaRef,
    ) -> DFResult<Self> {
        let schema = self.table.metadata().current_schema();
        let mut new_field_ids: Vec<i32> = Vec::with_capacity(dotted_paths.len());
        for path in &dotted_paths {
            let field_id = schema.field_id_by_name(path).ok_or_else(|| {
                datafusion::error::DataFusionError::Plan(format!(
                    "Column `{path}` not found in iceberg table schema"
                ))
            })?;
            new_field_ids.push(field_id);
        }

        let new_tasks: Vec<FileScanTask> = self
            .file_scan_tasks
            .iter()
            .map(|task| {
                let mut next = task.clone();
                next.project_field_ids = new_field_ids.clone();
                next
            })
            .collect();

        let plan_properties = Self::compute_properties(narrowed_schema, new_tasks.len());

        Ok(Self {
            table: self.table.clone(),
            snapshot_id: self.snapshot_id,
            plan_properties,
            projection: Some(dotted_paths),
            predicates: self.predicates.clone(),
            limit: self.limit,
            file_scan_tasks: new_tasks.into(),
        })
    }

    /// Computes [`PlanProperties`] used in query optimization.
    ///
    /// `n_files` is the number of [`FileScanTask`]s the planner produced;
    /// it determines how many partitions we expose. An empty file set
    /// still claims a single partition so `execute(0)` is always valid.
    fn compute_properties(schema: ArrowSchemaRef, n_files: usize) -> Arc<PlanProperties> {
        let partitions = n_files.max(1);
        Arc::new(PlanProperties::new(
            // TODO: declare lex ordering from default_sort_order and
            // partition-column constants from predicates.
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(partitions),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ))
    }
}

impl ExecutionPlan for IcebergTableScan {
    fn name(&self) -> &str {
        "IcebergTableScan"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan + 'static>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.plan_properties
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        // Empty scan: one virtual partition, zero rows.
        if self.file_scan_tasks.is_empty() {
            if partition != 0 {
                return Err(datafusion::error::DataFusionError::Execution(format!(
                    "IcebergTableScan: requested partition {partition} but scan is empty"
                )));
            }
            return Ok(Box::pin(RecordBatchStreamAdapter::new(
                self.schema(),
                futures::stream::empty(),
            )));
        }

        let task = self.file_scan_tasks[partition].clone();
        let file_io = self.table.file_io().clone();
        let fut = stream_one_file(file_io, task);
        let stream = futures::stream::once(fut).try_flatten();

        // Apply limit only when there's a single partition. With multiple
        // partitions a per-partition row count is the wrong shape for a
        // table-level limit, so we leave it to the DataFusion planner to
        // insert a `LimitExec` above the merge.
        let stream: futures::stream::BoxStream<'static, DFResult<RecordBatch>> =
            if self.file_scan_tasks.len() == 1 && self.limit.is_some() {
                let mut remaining = self.limit.unwrap();
                Box::pin(stream.try_filter_map(move |batch| {
                    futures::future::ready(if remaining == 0 {
                        Ok(None)
                    } else if batch.num_rows() <= remaining {
                        remaining -= batch.num_rows();
                        Ok(Some(batch))
                    } else {
                        let limited = batch.slice(0, remaining);
                        remaining = 0;
                        Ok(Some(limited))
                    })
                }))
            } else {
                Box::pin(stream)
            };

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream,
        )))
    }
}

impl DisplayAs for IcebergTableScan {
    fn fmt_as(
        &self,
        _t: datafusion::physical_plan::DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        write!(
            f,
            "IcebergTableScan projection:[{}] predicate:[{}]",
            self.projection
                .clone()
                .map_or(String::new(), |v| v.join(",")),
            self.predicates
                .clone()
                .map_or(String::from(""), |p| format!("{p}"))
        )?;
        if let Some(limit) = self.limit {
            write!(f, " limit:[{limit}]")?;
        }
        Ok(())
    }
}

/// Run iceberg's file planner once and materialise the surviving
/// [`FileScanTask`]s. Done eagerly in [`IcebergTableScan::try_new`] so
/// we know the partition count before computing `PlanProperties`.
async fn plan_files(
    table: &Table,
    snapshot_id: Option<i64>,
    column_names: Option<Vec<String>>,
    predicates: Option<Predicate>,
) -> DFResult<Arc<[FileScanTask]>> {
    let scan_builder = match snapshot_id {
        Some(snapshot_id) => table.scan().snapshot_id(snapshot_id),
        None => table.scan(),
    };

    let mut scan_builder = match column_names {
        Some(column_names) => scan_builder.select(column_names),
        None => scan_builder.select_all(),
    };
    if let Some(pred) = predicates {
        scan_builder = scan_builder.with_filter(pred);
    }
    let table_scan = scan_builder.build().map_err(to_datafusion_error)?;

    let file_stream = table_scan.plan_files().await.map_err(to_datafusion_error)?;
    let tasks: Vec<FileScanTask> = file_stream
        .try_collect()
        .await
        .map_err(to_datafusion_error)?;
    Ok(tasks.into())
}

/// Open one [`FileScanTask`] and adapt its row stream into a DataFusion
/// error type. Preserves the file's intrinsic ordering (the table's
/// `write.sort.order` applied to this file's contents).
async fn stream_one_file(
    file_io: iceberg::io::FileIO,
    task: FileScanTask,
) -> DFResult<impl Stream<Item = DFResult<RecordBatch>> + Send> {
    let reader = ArrowReaderBuilder::new(file_io).build();
    let stream = reader
        .stream_file(task)
        .await
        .map_err(to_datafusion_error)?;
    Ok(stream.map_err(to_datafusion_error))
}

fn get_column_names(
    schema: ArrowSchemaRef,
    projection: Option<&Vec<usize>>,
) -> Option<Vec<String>> {
    projection.map(|v| {
        v.iter()
            .map(|p| schema.field(*p).name().clone())
            .collect::<Vec<String>>()
    })
}
