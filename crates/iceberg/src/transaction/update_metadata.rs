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

use std::sync::Arc;

use async_trait::async_trait;

use crate::spec::{Schema, SchemaId};
use crate::table::Table;
use crate::transaction::action::{ActionCommit, TransactionAction};
use crate::{Error, ErrorKind, Result, TableRequirement, TableUpdate};

/// A transaction action for checking and updating table schema metadata.
///
/// This action exposes low-level schema metadata operations that map directly to
/// [`TableRequirement::CurrentSchemaIdMatch`], [`TableUpdate::AddSchema`], and
/// [`TableUpdate::SetCurrentSchema`]. Updates are replayed in insertion order when a transaction
/// is retried against refreshed table metadata.
#[derive(Debug, Default)]
pub struct UpdateMetadataAction {
    updates: Vec<TableUpdate>,
    requirements: Vec<TableRequirement>,
}

impl UpdateMetadataAction {
    /// Creates a new [`UpdateMetadataAction`] with no updates or requirements.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a requirement that the table's current schema id still matches `current_schema_id`.
    ///
    /// This is useful for optimistic concurrency control when subsequent updates depend on the
    /// table still using a specific current schema.
    pub fn check_current_schema(mut self, current_schema_id: SchemaId) -> Self {
        self.requirements
            .push(TableRequirement::CurrentSchemaIdMatch { current_schema_id });
        self
    }

    /// Adds a schema update to this action.
    ///
    /// Updates are applied in the same order they are added.
    pub fn add_schema(mut self, schema: Schema) -> Self {
        self.updates.push(TableUpdate::AddSchema { schema });
        self
    }

    /// Sets the table's current schema id.
    ///
    /// Passing `-1` selects the last schema added by this action.
    pub fn set_current_schema(mut self, schema_id: SchemaId) -> Self {
        self.updates
            .push(TableUpdate::SetCurrentSchema { schema_id });
        self
    }
}

#[async_trait]
impl TransactionAction for UpdateMetadataAction {
    async fn commit(self: Arc<Self>, _table: &Table) -> Result<ActionCommit> {
        if self.updates.is_empty() && self.requirements.is_empty() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "At least one current schema check or schema metadata update is required for UpdateMetadataAction",
            ));
        }

        Ok(ActionCommit::new(
            self.updates.clone(),
            self.requirements.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use as_any::Downcast;

    use crate::spec::Schema;
    use crate::transaction::tests::make_v2_table;
    use crate::transaction::update_metadata::UpdateMetadataAction;
    use crate::transaction::{ApplyTransactionAction, Transaction, TransactionAction};
    use crate::{ErrorKind, TableRequirement, TableUpdate};

    #[test]
    fn test_update_metadata_action() {
        let table = make_v2_table();
        let tx = Transaction::new(&table);
        let requirement = TableRequirement::CurrentSchemaIdMatch {
            current_schema_id: table.metadata().current_schema_id(),
        };
        let schema = Schema::builder().with_schema_id(7).build().unwrap();
        let updates = vec![
            TableUpdate::AddSchema {
                schema: schema.clone(),
            },
            TableUpdate::SetCurrentSchema { schema_id: -1 },
        ];

        let tx = tx
            .update_metadata()
            .check_current_schema(table.metadata().current_schema_id())
            .add_schema(schema)
            .set_current_schema(-1)
            .apply(tx)
            .unwrap();

        assert_eq!(tx.actions.len(), 1);

        let action = (*tx.actions[0])
            .downcast_ref::<UpdateMetadataAction>()
            .unwrap();

        assert_eq!(action.updates, updates);
        assert_eq!(action.requirements, vec![requirement]);
    }

    #[tokio::test]
    async fn test_update_metadata_commit() {
        let table = make_v2_table();
        let requirement = TableRequirement::CurrentSchemaIdMatch {
            current_schema_id: table.metadata().current_schema_id(),
        };
        let schema = Schema::builder().with_schema_id(7).build().unwrap();
        let updates = vec![
            TableUpdate::AddSchema {
                schema: schema.clone(),
            },
            TableUpdate::SetCurrentSchema { schema_id: -1 },
        ];

        let mut action_commit = Arc::new(
            UpdateMetadataAction::new()
                .check_current_schema(table.metadata().current_schema_id())
                .add_schema(schema)
                .set_current_schema(-1),
        )
        .commit(&table)
        .await
        .unwrap();

        assert_eq!(action_commit.take_updates(), updates);
        assert_eq!(action_commit.take_requirements(), vec![requirement]);
    }

    #[tokio::test]
    async fn test_update_metadata_check_current_schema_only() {
        let table = make_v2_table();
        let requirement = TableRequirement::CurrentSchemaIdMatch {
            current_schema_id: table.metadata().current_schema_id(),
        };

        let mut action_commit = Arc::new(
            UpdateMetadataAction::new().check_current_schema(table.metadata().current_schema_id()),
        )
        .commit(&table)
        .await
        .unwrap();

        assert!(action_commit.take_updates().is_empty());
        assert_eq!(action_commit.take_requirements(), vec![requirement]);
    }

    #[tokio::test]
    async fn test_update_metadata_commit_requires_check_or_update() {
        let table = make_v2_table();
        let result = Arc::new(UpdateMetadataAction::new()).commit(&table).await;
        assert!(result.is_err());
        let err = result.err().unwrap();

        assert_eq!(err.kind(), ErrorKind::DataInvalid);
    }
}
