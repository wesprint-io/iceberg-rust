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

//! Evaluates Parquet row-group bloom filters against scan predicates.
//!
//! Bloom filters can rule out a row group only for equality (`col = X`) and
//! `IN` predicates: a `false` from `Sbbf::check` means the value is
//! *definitely not* present. For every other operator, the bloom filter
//! cannot help and we conservatively return [`ROW_GROUP_MIGHT_MATCH`].

use std::collections::HashMap;

use fnv::FnvHashSet;
use parquet::bloom_filter::Sbbf;
use parquet::schema::types::{ColumnDescPtr, ColumnDescriptor};

use crate::Result;
use crate::expr::visitors::bound_predicate_visitor::{BoundPredicateVisitor, visit};
use crate::expr::{BoundPredicate, BoundReference};
use crate::spec::{Datum, PrimitiveLiteral};

const ROW_GROUP_MIGHT_MATCH: Result<bool> = Ok(true);
const ROW_GROUP_CANT_MATCH: Result<bool> = Ok(false);

/// Pre-loaded bloom filters for a single row group, keyed by parquet leaf
/// column index.
pub(crate) type RowGroupBloomFilters = HashMap<usize, Sbbf>;

pub(crate) struct RowGroupBloomFilterEvaluator<'a> {
    /// Bloom filters that have been loaded for this row group, keyed by the
    /// parquet leaf column index they cover.
    bloom_filters: &'a RowGroupBloomFilters,
    /// Iceberg field id → parquet leaf column index.
    field_id_map: &'a HashMap<i32, usize>,
    /// Parquet schema descriptors, used to encode literal values into the
    /// physical byte representation parquet uses for bloom-filter hashing.
    parquet_columns: &'a [ColumnDescPtr],
}

impl<'a> RowGroupBloomFilterEvaluator<'a> {
    pub(crate) fn new(
        bloom_filters: &'a RowGroupBloomFilters,
        field_id_map: &'a HashMap<i32, usize>,
        parquet_columns: &'a [ColumnDescPtr],
    ) -> Self {
        Self {
            bloom_filters,
            field_id_map,
            parquet_columns,
        }
    }

    /// Returns `Ok(true)` when the row group might contain rows matching
    /// `predicate` and `Ok(false)` when at least one bloom filter proves it
    /// cannot.
    pub(crate) fn eval(
        predicate: &BoundPredicate,
        bloom_filters: &'a RowGroupBloomFilters,
        field_id_map: &'a HashMap<i32, usize>,
        parquet_columns: &'a [ColumnDescPtr],
    ) -> Result<bool> {
        let mut evaluator = Self::new(bloom_filters, field_id_map, parquet_columns);
        visit(&mut evaluator, predicate)
    }

    /// Returns the bloom filter and column descriptor for `field_id` when both
    /// are present. `None` means we cannot make a statement and the caller
    /// should fall back to [`ROW_GROUP_MIGHT_MATCH`].
    fn bloom_filter_for(&self, field_id: i32) -> Option<(&Sbbf, &ColumnDescriptor)> {
        let column_idx = *self.field_id_map.get(&field_id)?;
        let sbbf = self.bloom_filters.get(&column_idx)?;
        let column_descr = self.parquet_columns.get(column_idx)?.as_ref();
        Some((sbbf, column_descr))
    }

    fn datum_might_be_in(&self, sbbf: &Sbbf, column_descr: &ColumnDescriptor, datum: &Datum) -> bool {
        match datum_to_bloom_filter_bytes(datum, column_descr) {
            Some(bytes) => {
                let result = sbbf.check(bytes.as_slice());
                if !result && std::env::var("ICEBERG_BLOOM_DEBUG").is_ok() {
                    eprintln!(
                        "[iceberg::bloom] CHECK_FALSE column={:?} parquet_path={:?} physical={:?} type_length={} datum={:?} bytes_hex={}",
                        column_descr.name(),
                        column_descr.path(),
                        column_descr.physical_type(),
                        column_descr.type_length(),
                        datum,
                        bytes.iter().map(|b| format!("{b:02x}")).collect::<String>(),
                    );
                }
                result
            }
            // Unsupported encoding: be conservative and assume the value
            // *may* be present.
            None => {
                if std::env::var("ICEBERG_BLOOM_DEBUG").is_ok() {
                    eprintln!(
                        "[iceberg::bloom] ENCODING_UNSUPPORTED column={:?} physical={:?} datum={:?}; treating as MIGHT_MATCH",
                        column_descr.name(),
                        column_descr.physical_type(),
                        datum,
                    );
                }
                true
            }
        }
    }
}

impl BoundPredicateVisitor for RowGroupBloomFilterEvaluator<'_> {
    type T = bool;

    fn always_true(&mut self) -> Result<bool> {
        ROW_GROUP_MIGHT_MATCH
    }

    fn always_false(&mut self) -> Result<bool> {
        ROW_GROUP_CANT_MATCH
    }

    fn and(&mut self, lhs: bool, rhs: bool) -> Result<bool> {
        Ok(lhs && rhs)
    }

    fn or(&mut self, lhs: bool, rhs: bool) -> Result<bool> {
        Ok(lhs || rhs)
    }

    fn not(&mut self, _inner: bool) -> Result<bool> {
        // Bloom filters can confirm non-presence (`check == false`) but never
        // confirm absence of *every* other value, so they cannot prune `NOT`
        // sub-trees.
        ROW_GROUP_MIGHT_MATCH
    }

    fn is_null(
        &mut self,
        _reference: &BoundReference,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        ROW_GROUP_MIGHT_MATCH
    }

    fn not_null(
        &mut self,
        _reference: &BoundReference,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        ROW_GROUP_MIGHT_MATCH
    }

    fn is_nan(
        &mut self,
        _reference: &BoundReference,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        ROW_GROUP_MIGHT_MATCH
    }

    fn not_nan(
        &mut self,
        _reference: &BoundReference,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        ROW_GROUP_MIGHT_MATCH
    }

    fn less_than(
        &mut self,
        _reference: &BoundReference,
        _datum: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        ROW_GROUP_MIGHT_MATCH
    }

    fn less_than_or_eq(
        &mut self,
        _reference: &BoundReference,
        _datum: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        ROW_GROUP_MIGHT_MATCH
    }

    fn greater_than(
        &mut self,
        _reference: &BoundReference,
        _datum: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        ROW_GROUP_MIGHT_MATCH
    }

    fn greater_than_or_eq(
        &mut self,
        _reference: &BoundReference,
        _datum: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        ROW_GROUP_MIGHT_MATCH
    }

    fn eq(
        &mut self,
        reference: &BoundReference,
        datum: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        match self.bloom_filter_for(reference.field().id) {
            Some((sbbf, column_descr)) => {
                if self.datum_might_be_in(sbbf, column_descr, datum) {
                    ROW_GROUP_MIGHT_MATCH
                } else {
                    ROW_GROUP_CANT_MATCH
                }
            }
            None => ROW_GROUP_MIGHT_MATCH,
        }
    }

    fn not_eq(
        &mut self,
        _reference: &BoundReference,
        _datum: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        ROW_GROUP_MIGHT_MATCH
    }

    fn starts_with(
        &mut self,
        _reference: &BoundReference,
        _datum: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        // Bloom filters hash whole values, not prefixes.
        ROW_GROUP_MIGHT_MATCH
    }

    fn not_starts_with(
        &mut self,
        _reference: &BoundReference,
        _datum: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        ROW_GROUP_MIGHT_MATCH
    }

    fn r#in(
        &mut self,
        reference: &BoundReference,
        literals: &FnvHashSet<Datum>,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        let Some((sbbf, column_descr)) = self.bloom_filter_for(reference.field().id) else {
            return ROW_GROUP_MIGHT_MATCH;
        };

        // The row group can only be skipped when *every* literal is proven
        // absent by the bloom filter.
        let any_might_match = literals
            .iter()
            .any(|datum| self.datum_might_be_in(sbbf, column_descr, datum));

        if any_might_match {
            ROW_GROUP_MIGHT_MATCH
        } else {
            ROW_GROUP_CANT_MATCH
        }
    }

    fn not_in(
        &mut self,
        _reference: &BoundReference,
        _literals: &FnvHashSet<Datum>,
        _predicate: &BoundPredicate,
    ) -> Result<bool> {
        ROW_GROUP_MIGHT_MATCH
    }
}

/// Walks a bound predicate and reports the parquet leaf column indices that
/// participate in equality / IN sub-predicates — the only ones whose row
/// groups could be pruned by a bloom filter. Used by the reader to decide
/// which (row_group, column) bloom filters are worth fetching.
pub(crate) fn collect_bloom_filterable_column_indices(
    predicate: &BoundPredicate,
    field_id_map: &HashMap<i32, usize>,
) -> Result<FnvHashSet<usize>> {
    let mut visitor = BloomFilterableColumnCollector {
        field_id_map,
        column_indices: FnvHashSet::default(),
    };
    visit(&mut visitor, predicate)?;
    Ok(visitor.column_indices)
}

struct BloomFilterableColumnCollector<'a> {
    field_id_map: &'a HashMap<i32, usize>,
    column_indices: FnvHashSet<usize>,
}

impl BloomFilterableColumnCollector<'_> {
    fn record(&mut self, reference: &BoundReference) {
        if let Some(idx) = self.field_id_map.get(&reference.field().id) {
            self.column_indices.insert(*idx);
        }
    }
}

impl BoundPredicateVisitor for BloomFilterableColumnCollector<'_> {
    type T = ();

    fn always_true(&mut self) -> Result<()> {
        Ok(())
    }

    fn always_false(&mut self) -> Result<()> {
        Ok(())
    }

    fn and(&mut self, _lhs: (), _rhs: ()) -> Result<()> {
        Ok(())
    }

    fn or(&mut self, _lhs: (), _rhs: ()) -> Result<()> {
        Ok(())
    }

    fn not(&mut self, _inner: ()) -> Result<()> {
        Ok(())
    }

    fn is_null(&mut self, _r: &BoundReference, _p: &BoundPredicate) -> Result<()> {
        Ok(())
    }

    fn not_null(&mut self, _r: &BoundReference, _p: &BoundPredicate) -> Result<()> {
        Ok(())
    }

    fn is_nan(&mut self, _r: &BoundReference, _p: &BoundPredicate) -> Result<()> {
        Ok(())
    }

    fn not_nan(&mut self, _r: &BoundReference, _p: &BoundPredicate) -> Result<()> {
        Ok(())
    }

    fn less_than(&mut self, _r: &BoundReference, _d: &Datum, _p: &BoundPredicate) -> Result<()> {
        Ok(())
    }

    fn less_than_or_eq(
        &mut self,
        _r: &BoundReference,
        _d: &Datum,
        _p: &BoundPredicate,
    ) -> Result<()> {
        Ok(())
    }

    fn greater_than(&mut self, _r: &BoundReference, _d: &Datum, _p: &BoundPredicate) -> Result<()> {
        Ok(())
    }

    fn greater_than_or_eq(
        &mut self,
        _r: &BoundReference,
        _d: &Datum,
        _p: &BoundPredicate,
    ) -> Result<()> {
        Ok(())
    }

    fn eq(&mut self, reference: &BoundReference, _d: &Datum, _p: &BoundPredicate) -> Result<()> {
        self.record(reference);
        Ok(())
    }

    fn not_eq(&mut self, _r: &BoundReference, _d: &Datum, _p: &BoundPredicate) -> Result<()> {
        Ok(())
    }

    fn starts_with(
        &mut self,
        _r: &BoundReference,
        _d: &Datum,
        _p: &BoundPredicate,
    ) -> Result<()> {
        Ok(())
    }

    fn not_starts_with(
        &mut self,
        _r: &BoundReference,
        _d: &Datum,
        _p: &BoundPredicate,
    ) -> Result<()> {
        Ok(())
    }

    fn r#in(
        &mut self,
        reference: &BoundReference,
        _literals: &FnvHashSet<Datum>,
        _p: &BoundPredicate,
    ) -> Result<()> {
        self.record(reference);
        Ok(())
    }

    fn not_in(
        &mut self,
        _r: &BoundReference,
        _literals: &FnvHashSet<Datum>,
        _p: &BoundPredicate,
    ) -> Result<()> {
        Ok(())
    }
}

/// Encodes an iceberg [`Datum`] into the physical byte representation parquet
/// uses for the column's bloom-filter hash. Returns `None` when the
/// (literal type, parquet physical type) pair is unsupported — callers must
/// then fall back to assuming the value *may* be present.
///
/// The bloom filter is computed by the writer as
/// `xxhash64(value.as_bytes())` where `value` is the parquet primitive
/// (`i32`, `i64`, `f32`, `f64`, `ByteArray`, `FixedLenByteArray`, …).
/// `as_bytes` for the integer types reinterprets the in-memory representation,
/// which on every platform iceberg-rust supports is little-endian; we
/// reproduce that exactly with `to_ne_bytes`.
fn datum_to_bloom_filter_bytes(datum: &Datum, column_descr: &ColumnDescriptor) -> Option<Vec<u8>> {
    use parquet::basic::Type as ParquetPhysicalType;

    match (datum.literal(), column_descr.physical_type()) {
        (PrimitiveLiteral::Boolean(value), ParquetPhysicalType::BOOLEAN) => {
            Some(vec![u8::from(*value)])
        }

        // Native ints — both writer and reader interpret `as_bytes` via raw
        // pointer cast, which equals `to_ne_bytes`.
        (PrimitiveLiteral::Int(value), ParquetPhysicalType::INT32) => {
            Some(value.to_ne_bytes().to_vec())
        }
        (PrimitiveLiteral::Long(value), ParquetPhysicalType::INT64) => {
            Some(value.to_ne_bytes().to_vec())
        }
        (PrimitiveLiteral::Float(value), ParquetPhysicalType::FLOAT) => {
            Some(value.into_inner().to_ne_bytes().to_vec())
        }
        (PrimitiveLiteral::Double(value), ParquetPhysicalType::DOUBLE) => {
            Some(value.into_inner().to_ne_bytes().to_vec())
        }

        // String / Binary as variable-length BYTE_ARRAY.
        (PrimitiveLiteral::String(value), ParquetPhysicalType::BYTE_ARRAY) => {
            Some(value.as_bytes().to_vec())
        }
        (PrimitiveLiteral::Binary(value), ParquetPhysicalType::BYTE_ARRAY) => Some(value.clone()),

        // Decimals can land in INT32 / INT64 / FIXED_LEN_BYTE_ARRAY depending
        // on precision. The mantissa is a signed integer in all three cases.
        (PrimitiveLiteral::Int128(mantissa), ParquetPhysicalType::INT32) => {
            let value: i32 = i32::try_from(*mantissa).ok()?;
            Some(value.to_ne_bytes().to_vec())
        }
        (PrimitiveLiteral::Int128(mantissa), ParquetPhysicalType::INT64) => {
            let value: i64 = i64::try_from(*mantissa).ok()?;
            Some(value.to_ne_bytes().to_vec())
        }
        (PrimitiveLiteral::Int128(mantissa), ParquetPhysicalType::FIXED_LEN_BYTE_ARRAY) => {
            let length = column_descr.type_length();
            if length <= 0 || length as usize > 16 {
                return None;
            }
            let length = length as usize;
            let be_bytes = mantissa.to_be_bytes();
            Some(be_bytes[16 - length..].to_vec())
        }

        // Fixed-length binary (uuid / iceberg `fixed`) → already raw bytes.
        (PrimitiveLiteral::Binary(value), ParquetPhysicalType::FIXED_LEN_BYTE_ARRAY) => {
            let length = column_descr.type_length();
            if length <= 0 || value.len() != length as usize {
                return None;
            }
            Some(value.clone())
        }

        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use parquet::bloom_filter::Sbbf;
    use parquet::data_type::{ByteArray, FixedLenByteArray};
    use parquet::schema::parser::parse_message_type;
    use parquet::schema::types::SchemaDescriptor;

    use super::{
        RowGroupBloomFilterEvaluator, collect_bloom_filterable_column_indices,
        datum_to_bloom_filter_bytes,
    };
    use crate::expr::{Bind, Reference};
    use crate::spec::{Datum, NestedField, PrimitiveType, Schema, SchemaRef, Type};

    fn parquet_schema_with_nested_decimal_value() -> SchemaDescriptor {
        // nested.value is a decimal(20, 0) stored as a 9-byte fixed-len
        // byte array, mirroring what parquet writes for a 20-digit decimal.
        let message_type = "
message schema {
  required int32 id = 1;
  required group nested = 2 {
    required fixed_len_byte_array(9) value (DECIMAL(20,0)) = 3;
  }
}
        ";
        let parquet_type = parse_message_type(message_type).expect("parse message type");
        SchemaDescriptor::new(Arc::new(parquet_type))
    }

    fn iceberg_schema_with_nested_decimal_value() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(
                        2,
                        "nested",
                        Type::Struct(crate::spec::StructType::new(vec![
                            NestedField::required(
                                3,
                                "value",
                                Type::Primitive(PrimitiveType::Decimal {
                                    precision: 20,
                                    scale: 0,
                                }),
                            )
                            .into(),
                        ])),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        )
    }

    #[test]
    fn datum_encoding_matches_parquet_writer_for_decimal_flb() {
        let schema = parquet_schema_with_nested_decimal_value();
        let value_descr = schema
            .columns()
            .iter()
            .find(|col| col.name() == "value")
            .expect("value column");

        // Parquet writes Decimal128 as the 9 trailing big-endian bytes of the
        // i128 mantissa for a 20-digit decimal.
        let mantissa: i128 = 12_345_678_901_234_567_890_i128;
        let datum = Datum::decimal_with_precision(
            mantissa.to_string().parse().expect("parse decimal"),
            20,
        )
        .expect("build decimal datum");

        let our_bytes = datum_to_bloom_filter_bytes(&datum, value_descr)
            .expect("encoder should support decimal(20,0) FLB");

        let writer_bytes = mantissa.to_be_bytes()[16 - 9..].to_vec();
        assert_eq!(our_bytes, writer_bytes);
    }

    #[test]
    fn datum_encoding_supports_native_ints_and_strings() {
        let schema_str = "
message schema {
  required int32 i32_col = 1;
  required int64 i64_col = 2;
  required binary str_col (STRING) = 3;
}
        ";
        let parquet_type = parse_message_type(schema_str).expect("parse");
        let schema = SchemaDescriptor::new(Arc::new(parquet_type));
        let i32_descr = &schema.columns()[0];
        let i64_descr = &schema.columns()[1];
        let str_descr = &schema.columns()[2];

        assert_eq!(
            datum_to_bloom_filter_bytes(&Datum::int(42), i32_descr).expect("i32"),
            42_i32.to_ne_bytes().to_vec(),
        );
        assert_eq!(
            datum_to_bloom_filter_bytes(&Datum::long(42_i64), i64_descr).expect("i64"),
            42_i64.to_ne_bytes().to_vec(),
        );
        assert_eq!(
            datum_to_bloom_filter_bytes(&Datum::string("hello"), str_descr).expect("string"),
            b"hello".to_vec(),
        );
    }

    #[test]
    fn evaluator_skips_row_group_when_bloom_filter_definitely_excludes_value() {
        let schema = iceberg_schema_with_nested_decimal_value();
        let parquet_schema = parquet_schema_with_nested_decimal_value();
        let value_column_idx = parquet_schema
            .columns()
            .iter()
            .position(|col| col.name() == "value")
            .expect("value column index");

        // Build a bloom filter that contains value = 1234 but not 9999.
        let mut sbbf = Sbbf::new_with_ndv_fpp(8, 0.01).expect("build sbbf");
        let value_descr = &parquet_schema.columns()[value_column_idx];
        let present_bytes = datum_to_bloom_filter_bytes(
            &Datum::decimal_with_precision("1234".parse().expect("parse"), 20).unwrap(),
            value_descr,
        )
        .unwrap();
        sbbf.insert(&FixedLenByteArray::from(ByteArray::from(present_bytes)));

        let mut bloom_filters = HashMap::new();
        bloom_filters.insert(value_column_idx, sbbf);

        let mut field_id_map = HashMap::new();
        // id (field 1) -> parquet column 0
        field_id_map.insert(1_i32, 0);
        // nested.value (field 3) -> parquet column 1
        field_id_map.insert(3_i32, value_column_idx);

        let parquet_columns = parquet_schema.columns();

        // Predicate that the bloom filter can prove false.
        let predicate = Reference::new("nested.value")
            .equal_to(Datum::decimal_with_precision("9999".parse().unwrap(), 20).unwrap());
        let bound = predicate.bind(schema.clone(), true).unwrap();
        let result =
            RowGroupBloomFilterEvaluator::eval(&bound, &bloom_filters, &field_id_map, parquet_columns)
                .unwrap();
        assert!(!result, "9999 not in bloom -> row group should be skipped");

        // Predicate that the bloom filter cannot disprove (1234 is present).
        let predicate_present = Reference::new("nested.value")
            .equal_to(Datum::decimal_with_precision("1234".parse().unwrap(), 20).unwrap());
        let bound_present = predicate_present.bind(schema.clone(), true).unwrap();
        let result_present = RowGroupBloomFilterEvaluator::eval(
            &bound_present,
            &bloom_filters,
            &field_id_map,
            parquet_columns,
        )
        .unwrap();
        assert!(
            result_present,
            "1234 in bloom -> row group must be kept (no false negatives)"
        );
    }

    #[test]
    fn collector_records_eq_and_in_columns_only() {
        let schema = iceberg_schema_with_nested_decimal_value();
        let mut field_id_map = HashMap::new();
        field_id_map.insert(1_i32, 0);
        field_id_map.insert(3_i32, 1);

        // nested.value = 5 AND id > 0 — only the equality should land in
        // the bloom-filterable set.
        let predicate = Reference::new("nested.value")
            .equal_to(Datum::decimal_with_precision("5".parse().unwrap(), 20).unwrap())
            .and(Reference::new("id").greater_than(Datum::int(0)));
        let bound = predicate.bind(schema, true).unwrap();

        let collected =
            collect_bloom_filterable_column_indices(&bound, &field_id_map).expect("collect");
        assert_eq!(collected.len(), 1);
        assert!(collected.contains(&1_usize));
    }
}
