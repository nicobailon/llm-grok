"""Centralized constants for the llm-grok package."""

# Default configuration values
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_CONNECTIONS = 10
DEFAULT_KEEPALIVE_RATIO = 0.5
DEFAULT_TEMPERATURE = 0.0
TEMPERATURE_MIN = 0
TEMPERATURE_MAX = 1

# Network and streaming constants
HTTP_CHUNK_SIZE = 8192
SLEEP_INTERVAL_MAX = 0.5
DEFAULT_ENCODING = "utf-8"
SSE_DELIMITER = "\n\n"
SSE_EVENT_PREFIX = "event: "
SSE_DATA_PREFIX = "data: "
SSE_EVENT_PREFIX_LENGTH = len(SSE_EVENT_PREFIX)
SSE_DATA_PREFIX_LENGTH = len(SSE_DATA_PREFIX)

# Retry and backoff constants
JITTER_FACTOR_MIN = 0.1
JITTER_FACTOR_MAX = 0.25
DEFAULT_RETRY_DELAY = 1.0  # Base delay for retry attempts in seconds
MAX_RETRY_DELAY = 60.0     # Maximum delay for retry attempts in seconds

# Image processing constants
IMAGE_HEADER_BYTES = 16
WEBP_HEADER_BYTES = 20
WEBP_HEADER_CHECK_BYTES = 12
MIN_BASE64_IMAGE_LENGTH = 20
MAX_IMAGE_SIZE = 1 * 1024 * 1024  # 1MB
IMAGE_FETCH_TIMEOUT = 5

# Security constants
BLOCKED_PORTS = frozenset({22, 23, 25, 3306, 5432, 6379, 27017, 9200, 11211})
AWS_METADATA_ENDPOINT = '169.254.169.254'

# Response parsing constants
FIRST_CHOICE_INDEX = 0

# API endpoints
CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions"
MESSAGES_ENDPOINT = "/v1/messages"