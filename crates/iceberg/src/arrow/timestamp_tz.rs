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

//! Timestamp timezone normalization for Parquet files.

use std::sync::Arc;

use arrow_schema::{
    DataType, Field, Fields, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef,
};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use crate::arrow::schema::{
    ArrowSchemaVisitor, DEFAULT_MAP_FIELD_NAME, UTC_TIME_ZONE, visit_schema,
};
use crate::error::Result;
use crate::spec::{PrimitiveType, Schema, Type};
use crate::{Error, ErrorKind};

/// Coerce Arrow schema timezones for timestamp columns to the Iceberg-canonical form.
///
/// Iceberg's Arrow mapping renders `timestamptz` / `timestamptz_ns` as
/// `Timestamp(_, Some("+00:00"))` (see [`UTC_TIME_ZONE`]). arrow-rs's parquet reader
/// renders `TIMESTAMP(isAdjustedToUTC=true)` as `Timestamp(_, Some("UTC"))`. The two
/// timezones are semantically identical but compare unequal as Arrow `DataType`s,
/// which breaks downstream schema-equality checks (e.g. DataFusion's runtime batch
/// validation against an `IcebergStaticTableProvider::schema()`).
///
/// Walk the parquet-derived Arrow schema and, for any timestamp field whose Iceberg
/// counterpart is `Timestamptz`/`TimestamptzNs` and whose current zone is not already
/// [`UTC_TIME_ZONE`], rewrite the zone to [`UTC_TIME_ZONE`]. We use arrow-rs's schema
/// hint mechanism so the parquet reader emits batches with the canonical zone from
/// the start, matching what the rest of the codebase produces via
/// `schema_to_arrow_schema`.
///
/// Returns `None` when no field needed rewriting, mirroring [`coerce_int96_timestamps`].
pub(crate) fn coerce_timestamp_tz(
    arrow_schema: &ArrowSchemaRef,
    iceberg_schema: &Schema,
) -> Option<Arc<ArrowSchema>> {
    let mut visitor = TimestampTzCoercionVisitor::new(iceberg_schema);
    let coerced = visit_schema(arrow_schema, &mut visitor).ok()?;
    if visitor.changed {
        Some(Arc::new(coerced))
    } else {
        None
    }
}

/// Visitor that rewrites non-canonical Timestamp timezones (e.g. `"UTC"`) on fields
/// whose Iceberg counterpart is `Timestamptz`/`TimestamptzNs` to [`UTC_TIME_ZONE`].
struct TimestampTzCoercionVisitor<'a> {
    iceberg_schema: &'a Schema,
    // TODO(#2310): use FieldRef (Arc<Field>) once ArrowSchemaVisitor passes FieldRef.
    field_stack: Vec<Field>,
    changed: bool,
}

impl<'a> TimestampTzCoercionVisitor<'a> {
    fn new(iceberg_schema: &'a Schema) -> Self {
        Self {
            iceberg_schema,
            field_stack: Vec::new(),
            changed: false,
        }
    }

    /// If `field` is a `Timestamp(_, Some(zone))` whose Iceberg counterpart is
    /// `Timestamptz`/`TimestamptzNs` and `zone != UTC_TIME_ZONE`, return the
    /// rewritten data type; otherwise `None`.
    fn target_type(&self, field: &Field) -> Option<DataType> {
        let (unit, zone) = match field.data_type() {
            DataType::Timestamp(unit, Some(zone)) => (*unit, zone.clone()),
            _ => return None,
        };
        if zone.as_ref() == UTC_TIME_ZONE {
            return None;
        }

        let iceberg_is_timestamptz = field
            .metadata()
            .get(PARQUET_FIELD_ID_META_KEY)
            .and_then(|id_str| id_str.parse::<i32>().ok())
            .and_then(|field_id| self.iceberg_schema.field_by_id(field_id))
            .is_some_and(|f| {
                matches!(
                    &*f.field_type,
                    Type::Primitive(PrimitiveType::Timestamptz | PrimitiveType::TimestamptzNs)
                )
            });

        if iceberg_is_timestamptz {
            Some(DataType::Timestamp(unit, Some(UTC_TIME_ZONE.into())))
        } else {
            None
        }
    }
}

impl ArrowSchemaVisitor for TimestampTzCoercionVisitor<'_> {
    type T = Field;
    type U = ArrowSchema;

    fn before_field(&mut self, field: &Field) -> Result<()> {
        self.field_stack.push(field.as_ref().clone());
        Ok(())
    }

    fn after_field(&mut self, _field: &Field) -> Result<()> {
        self.field_stack.pop();
        Ok(())
    }

    fn before_list_element(&mut self, field: &Field) -> Result<()> {
        self.field_stack.push(field.as_ref().clone());
        Ok(())
    }

    fn after_list_element(&mut self, _field: &Field) -> Result<()> {
        self.field_stack.pop();
        Ok(())
    }

    fn before_map_key(&mut self, field: &Field) -> Result<()> {
        self.field_stack.push(field.as_ref().clone());
        Ok(())
    }

    fn after_map_key(&mut self, _field: &Field) -> Result<()> {
        self.field_stack.pop();
        Ok(())
    }

    fn before_map_value(&mut self, field: &Field) -> Result<()> {
        self.field_stack.push(field.as_ref().clone());
        Ok(())
    }

    fn after_map_value(&mut self, _field: &Field) -> Result<()> {
        self.field_stack.pop();
        Ok(())
    }

    fn schema(&mut self, schema: &ArrowSchema, values: Vec<Field>) -> Result<ArrowSchema> {
        Ok(ArrowSchema::new_with_metadata(
            values,
            schema.metadata().clone(),
        ))
    }

    fn r#struct(&mut self, _fields: &Fields, results: Vec<Field>) -> Result<Field> {
        let field_info = self
            .field_stack
            .last()
            .ok_or_else(|| Error::new(ErrorKind::Unexpected, "Field stack underflow in struct"))?;
        Ok(Field::new(
            field_info.name(),
            DataType::Struct(Fields::from(results)),
            field_info.is_nullable(),
        )
        .with_metadata(field_info.metadata().clone()))
    }

    fn list(&mut self, list: &DataType, value: Field) -> Result<Field> {
        let field_info = self
            .field_stack
            .last()
            .ok_or_else(|| Error::new(ErrorKind::Unexpected, "Field stack underflow in list"))?;
        let list_type = match list {
            DataType::List(_) => DataType::List(Arc::new(value)),
            DataType::LargeList(_) => DataType::LargeList(Arc::new(value)),
            DataType::FixedSizeList(_, size) => DataType::FixedSizeList(Arc::new(value), *size),
            _ => {
                return Err(Error::new(
                    ErrorKind::Unexpected,
                    format!("Expected list type, got {list}"),
                ));
            }
        };
        Ok(
            Field::new(field_info.name(), list_type, field_info.is_nullable())
                .with_metadata(field_info.metadata().clone()),
        )
    }

    fn map(&mut self, map: &DataType, key_value: Field, value: Field) -> Result<Field> {
        let field_info = self
            .field_stack
            .last()
            .ok_or_else(|| Error::new(ErrorKind::Unexpected, "Field stack underflow in map"))?;
        let sorted = match map {
            DataType::Map(_, sorted) => *sorted,
            _ => {
                return Err(Error::new(
                    ErrorKind::Unexpected,
                    format!("Expected map type, got {map}"),
                ));
            }
        };
        let struct_field = Field::new(
            DEFAULT_MAP_FIELD_NAME,
            DataType::Struct(Fields::from(vec![key_value, value])),
            false,
        );
        Ok(Field::new(
            field_info.name(),
            DataType::Map(Arc::new(struct_field), sorted),
            field_info.is_nullable(),
        )
        .with_metadata(field_info.metadata().clone()))
    }

    fn primitive(&mut self, p: &DataType) -> Result<Field> {
        let field_info = self.field_stack.last().ok_or_else(|| {
            Error::new(ErrorKind::Unexpected, "Field stack underflow in primitive")
        })?;

        if let Some(target_type) = self.target_type(field_info) {
            self.changed = true;
            Ok(
                Field::new(field_info.name(), target_type, field_info.is_nullable())
                    .with_metadata(field_info.metadata().clone()),
            )
        } else {
            Ok(
                Field::new(field_info.name(), p.clone(), field_info.is_nullable())
                    .with_metadata(field_info.metadata().clone()),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_schema::{DataType, Field, Schema as ArrowSchema, TimeUnit};
    use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

    use super::coerce_timestamp_tz;
    use crate::arrow::schema::UTC_TIME_ZONE;
    use crate::spec::{NestedField, PrimitiveType, Schema, Type};

    fn field_id_meta(id: i32) -> HashMap<String, String> {
        HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), id.to_string())])
    }

    fn iceberg_schema_with_timestamptz() -> Schema {
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::optional(1, "ts", Type::Primitive(PrimitiveType::Timestamptz)).into(),
                NestedField::required(2, "id", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap()
    }

    #[test]
    fn test_coerce_utc_to_canonical() {
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                true,
            )
            .with_metadata(field_id_meta(1)),
            Field::new("id", DataType::Int32, false).with_metadata(field_id_meta(2)),
        ]));

        let coerced =
            coerce_timestamp_tz(&arrow_schema, &iceberg_schema_with_timestamptz()).unwrap();
        assert_eq!(
            coerced.field(0).data_type(),
            &DataType::Timestamp(TimeUnit::Microsecond, Some(UTC_TIME_ZONE.into()))
        );
        // Non-timestamp field unchanged.
        assert_eq!(coerced.field(1).data_type(), &DataType::Int32);
    }

    #[test]
    fn test_no_change_when_already_canonical() {
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Microsecond, Some(UTC_TIME_ZONE.into())),
                true,
            )
            .with_metadata(field_id_meta(1)),
        ]));

        assert!(coerce_timestamp_tz(&arrow_schema, &iceberg_schema_with_timestamptz()).is_none());
    }

    #[test]
    fn test_no_change_when_iceberg_is_not_timestamptz() {
        // Iceberg says `timestamp` (no tz), arrow has a zoned timestamp — leave it alone.
        let iceberg = Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::optional(1, "ts", Type::Primitive(PrimitiveType::Timestamp)).into(),
            ])
            .build()
            .unwrap();

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                true,
            )
            .with_metadata(field_id_meta(1)),
        ]));

        assert!(coerce_timestamp_tz(&arrow_schema, &iceberg).is_none());
    }

    #[test]
    fn test_coerce_inside_struct() {
        // Mirrors the nested-projection case that triggered the original bug: a
        // `timestamptz` leaf inside a struct should be normalized to the canonical zone.
        let iceberg = Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::optional(
                    1,
                    "payload",
                    Type::Struct(crate::spec::StructType::new(vec![
                        NestedField::optional(2, "ts", Type::Primitive(PrimitiveType::Timestamptz))
                            .into(),
                    ])),
                )
                .into(),
            ])
            .build()
            .unwrap();

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new(
                "payload",
                DataType::Struct(
                    vec![
                        Field::new(
                            "ts",
                            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                            true,
                        )
                        .with_metadata(field_id_meta(2)),
                    ]
                    .into(),
                ),
                true,
            )
            .with_metadata(field_id_meta(1)),
        ]));

        let coerced = coerce_timestamp_tz(&arrow_schema, &iceberg).unwrap();
        let DataType::Struct(fields) = coerced.field(0).data_type() else {
            panic!("expected struct");
        };
        assert_eq!(
            fields[0].data_type(),
            &DataType::Timestamp(TimeUnit::Microsecond, Some(UTC_TIME_ZONE.into()))
        );
    }
}
