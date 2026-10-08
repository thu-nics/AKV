//! Router implementations

use std::fmt::Debug;

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};

use crate::protocols::{
    chat::ChatCompletionRequest,
    classify::ClassifyRequest,
    completion::CompletionRequest,
    embedding::EmbeddingRequest,
    generate::GenerateRequest,
    rerank::RerankRequest,
    responses::{ResponsesGetParams, ResponsesRequest},
};

pub mod conversations;
pub mod error;
pub mod factory;
pub mod grpc;
pub mod header_utils;
pub mod http;
pub mod mcp_utils;
pub mod mesh;
pub mod openai;
pub mod parse;
pub mod persistence_utils;
pub mod router_manager;
pub mod streaming_utils;
pub mod tokenize;

pub use factory::RouterFactory;
// Re-export HTTP routers for convenience
pub use http::{pd_router, pd_types, router};

/// Keep SRT extensions beside the upstream schema, retaining its validation.
/// Unknown fields remain ignored, as in the original OpenAI protocol extractor.
#[derive(Clone, serde::Deserialize, serde::Serialize)]
pub struct ExtendedChatRequest {
    #[serde(flatten)]
    pub request: ChatCompletionRequest,
    #[serde(flatten)]
    pub extensions: serde_json::Map<String, serde_json::Value>,
}

impl crate::protocols::validated::Normalizable for ExtendedChatRequest {
    fn normalize(&mut self) {
        crate::protocols::validated::Normalizable::normalize(&mut self.request);
        self.extensions.retain(|key, _| {
            matches!(
                key.as_str(),
                "drop_message" | "drop_rule" | "reposition" | "return_meta_info"
            )
        });
    }
}

impl validator::Validate for ExtendedChatRequest {
    fn validate(&self) -> Result<(), validator::ValidationErrors> {
        validator::Validate::validate(&self.request)
    }
}

impl crate::protocols::common::GenerationRequest for ExtendedChatRequest {
    fn is_stream(&self) -> bool {
        self.request.stream
    }
    fn get_model(&self) -> Option<&str> {
        Some(&self.request.model)
    }
    fn extract_text_for_routing(&self) -> String {
        crate::protocols::common::GenerationRequest::extract_text_for_routing(&self.request)
    }
}

/// Core trait for all router implementations
///
/// This trait provides a unified interface for routing requests,
/// regardless of whether it's a regular router or PD router.
#[async_trait]
pub trait RouterTrait: Send + Sync + Debug {
    /// Get a reference to self as Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;

    /// Route a health generate request
    async fn health_generate(&self, _req: Request<Body>) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Health generate not implemented",
        )
            .into_response()
    }

    /// Get server information
    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "Server info not implemented").into_response()
    }

    /// Get available models
    async fn get_models(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "Get models not implemented").into_response()
    }

    /// Get model information
    async fn get_model_info(&self, _req: Request<Body>) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Get model info not implemented",
        )
            .into_response()
    }

    /// Route a generate request
    async fn route_generate(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &GenerateRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Generate endpoint not implemented",
        )
            .into_response()
    }

    /// Route a chat completion request
    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response;

    /// Optional SRT fields use the same routing/dispatch pipeline. Backends
    /// without an implementation must reject features instead of losing them.
    async fn route_chat_with_extensions(
        &self,
        headers: Option<&HeaderMap>,
        body: &ExtendedChatRequest,
        model_id: Option<&str>,
    ) -> Response {
        if body
            .extensions
            .keys()
            .any(|key| matches!(key.as_str(), "drop_message" | "drop_rule" | "reposition"))
        {
            return error::bad_request(
                "unsupported_context",
                "This router does not support Drop/Reposition requests",
            );
        }
        self.route_chat(headers, &body.request, model_id).await
    }

    /// Route a completion request
    async fn route_completion(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &CompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Completion endpoint not implemented",
        )
            .into_response()
    }

    /// Route a responses request
    async fn route_responses(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &ResponsesRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Responses endpoint not implemented",
        )
            .into_response()
    }

    /// Retrieve a stored/background response by id
    async fn get_response(
        &self,
        _headers: Option<&HeaderMap>,
        _response_id: &str,
        _params: &ResponsesGetParams,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "Get response not implemented").into_response()
    }

    /// Cancel a background response by id
    async fn cancel_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Cancel response not implemented",
        )
            .into_response()
    }

    /// Delete a response by id
    async fn delete_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Responses delete endpoint not implemented",
        )
            .into_response()
    }

    /// List input items of a response by id
    async fn list_response_input_items(
        &self,
        _headers: Option<&HeaderMap>,
        _response_id: &str,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Responses list input items endpoint not implemented",
        )
            .into_response()
    }

    /// Route embedding requests (OpenAI-compatible /v1/embeddings)
    async fn route_embeddings(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &EmbeddingRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "Embeddings not implemented").into_response()
    }

    /// Route classification requests (OpenAI-compatible /v1/classify)
    async fn route_classify(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &ClassifyRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "Classify not implemented").into_response()
    }

    /// Route rerank requests
    async fn route_rerank(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &RerankRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "Rerank not implemented").into_response()
    }

    /// Get router type name
    fn router_type(&self) -> &'static str;

    /// Check if this is a PD router
    fn is_pd_mode(&self) -> bool {
        self.router_type() == "pd"
    }
}
