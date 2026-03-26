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

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{NestedField, PrimitiveType, Schema, Transform, Type, UnboundPartitionSpec};
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, Result, TableCreation, TableIdent};

fn base_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "foo", Type::Primitive(PrimitiveType::Int)).into(),
        ])
        .build()
        .unwrap()
}

fn second_schema() -> Schema {
    Schema::builder()
        .with_schema_id(10)
        .with_fields(vec![
            NestedField::required(1, "foo", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "bar", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .unwrap()
}

fn third_schema() -> Schema {
    Schema::builder()
        .with_schema_id(20)
        .with_fields(vec![
            NestedField::required(1, "foo", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "bar", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(3, "baz", Type::Primitive(PrimitiveType::Boolean)).into(),
        ])
        .build()
        .unwrap()
}

fn partition_spec() -> UnboundPartitionSpec {
    UnboundPartitionSpec::builder()
        .add_partition_field(1, "foo_identity", Transform::Identity)
        .unwrap()
        .build()
}

fn warehouse_path() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "iceberg_update_metadata_{}_{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[tokio::test]
async fn test_update_metadata_adds_multiple_schemas_and_sets_current_schema() -> Result<()> {
    let warehouse = warehouse_path();
    let catalog = MemoryCatalogBuilder::default()
        .load(
            "memory",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                warehouse.to_str().unwrap().to_string(),
            )]),
        )
        .await?;

    let namespace = NamespaceIdent::new("ns".into());
    catalog.create_namespace(&namespace, HashMap::new()).await?;

    let table_name = "tbl";
    let table_ident = TableIdent::new(namespace.clone(), table_name.to_string());
    let table = catalog
        .create_table(
            &namespace,
            TableCreation::builder()
                .name(table_name.to_string())
                .schema(base_schema())
                .build(),
        )
        .await?;

    let initial_current_schema_id = table.metadata().current_schema_id();

    let tx = Transaction::new(&table);
    let updated_table = tx
        .update_metadata()
        .check_current_schema(initial_current_schema_id)
        .add_schema(second_schema())
        .add_schema(third_schema())
        .set_current_schema(-1)
        .apply(tx)?;

    let updated_table = updated_table.commit(&catalog).await?;
    assert_eq!(updated_table.identifier(), &table_ident);

    let loaded_table = catalog.load_table(&table_ident).await?;
    let metadata = loaded_table.metadata();

    assert_eq!(metadata.schemas_iter().count(), 3);
    assert_ne!(metadata.current_schema_id(), initial_current_schema_id);

    let current_schema = metadata.current_schema();
    assert_eq!(current_schema.schema_id(), metadata.current_schema_id());
    assert!(current_schema.field_by_name("foo").is_some());
    assert!(current_schema.field_by_name("bar").is_some());
    assert!(current_schema.field_by_name("baz").is_some());

    let original_schema = metadata.schema_by_id(initial_current_schema_id).unwrap();
    assert!(original_schema.field_by_name("foo").is_some());
    assert!(original_schema.field_by_name("bar").is_none());
    assert!(original_schema.field_by_name("baz").is_none());

    let _ = std::fs::remove_dir_all(&warehouse);

    Ok(())
}

#[tokio::test]
async fn test_update_metadata_adds_partition_spec_and_sets_default_spec() -> Result<()> {
    let warehouse = warehouse_path();
    let catalog = MemoryCatalogBuilder::default()
        .load(
            "memory",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                warehouse.to_str().unwrap().to_string(),
            )]),
        )
        .await?;

    let namespace = NamespaceIdent::new("ns_partition".into());
    catalog.create_namespace(&namespace, HashMap::new()).await?;

    let table_name = "tbl_partition";
    let table_ident = TableIdent::new(namespace.clone(), table_name.to_string());
    let table = catalog
        .create_table(
            &namespace,
            TableCreation::builder()
                .name(table_name.to_string())
                .schema(base_schema())
                .build(),
        )
        .await?;

    let initial_current_schema_id = table.metadata().current_schema_id();
    let initial_default_spec_id = table.metadata().default_partition_spec_id();

    let tx = Transaction::new(&table);
    let updated_table = tx
        .update_metadata()
        .check_current_schema(initial_current_schema_id)
        .check_default_partition_spec(initial_default_spec_id)
        .add_partition_spec(partition_spec())
        .set_default_partition_spec(-1)
        .apply(tx)?;

    let updated_table = updated_table.commit(&catalog).await?;
    assert_eq!(updated_table.identifier(), &table_ident);

    let loaded_table = catalog.load_table(&table_ident).await?;
    let metadata = loaded_table.metadata();

    assert_eq!(metadata.partition_specs_iter().count(), 2);
    assert_ne!(
        metadata.default_partition_spec_id(),
        initial_default_spec_id
    );

    let default_spec = metadata.default_partition_spec();
    assert_eq!(default_spec.spec_id(), metadata.default_partition_spec_id());
    assert_eq!(default_spec.fields().len(), 1);
    assert_eq!(default_spec.fields()[0].source_id, 1);
    assert_eq!(default_spec.fields()[0].name, "foo_identity");
    assert_eq!(default_spec.fields()[0].transform, Transform::Identity);

    let original_spec = metadata
        .partition_spec_by_id(initial_default_spec_id)
        .unwrap();
    assert!(original_spec.fields().is_empty());

    let _ = std::fs::remove_dir_all(&warehouse);

    Ok(())
}
