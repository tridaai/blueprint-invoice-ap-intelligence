"""Object storage for document blobs: the S3-compatible port.

Supplier documents arrive as files — scanned invoice PDFs, email
attachments. In production those bytes live in object storage, and
the pipeline references them by **object key**, never by local path:
``DocumentInput.object_key`` names the blob, and the OCR node pulls
the bytes through this port when a document carries no text layer of
its own.

Implementations:

- ``S3ObjectStore`` — boto3 against any S3-compatible endpoint:
  AWS S3, or MinIO (the compose stack runs one for local use).
  Configured by ``S3_ENDPOINT_URL`` (blank = AWS), ``S3_BUCKET``,
  ``S3_ACCESS_KEY`` / ``S3_SECRET_KEY``, ``S3_REGION``. The bucket is
  created lazily on first write when it does not exist (the MinIO
  local flow); against AWS, pre-create the bucket as usual.
- ``InMemoryObjectStore`` — the test double.

``get_object_store`` returns None when no bucket is configured —
documents then must carry their own text layer (the samples do),
which is exactly the hermetic offline path.
"""

from __future__ import annotations

import threading

from .config import env_str, load_dotenv


class InMemoryObjectStore:
    """Process-local object store — a test double. Thread-safe."""

    name = "memory"

    def __init__(self) -> None:
        self._objects: dict[str, tuple[bytes, str]] = {}
        self._lock = threading.Lock()

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        with self._lock:
            self._objects[key] = (bytes(data), content_type)
        return key

    def get(self, key: str) -> bytes:
        with self._lock:
            if key not in self._objects:
                raise KeyError(f"No object stored under key: {key}")
            return self._objects[key][0]

    def exists(self, key: str) -> bool:
        with self._lock:
            return key in self._objects


class S3ObjectStore:
    """S3-compatible object store via boto3 (the ``s3`` extra)."""

    name = "s3"

    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        region: str = "us-east-1",
    ) -> None:
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "S3 object storage is configured but boto3 is not installed. "
                "Install the s3 extra: uv sync --extra s3 "
                "(pip fallback: pip install -e \".[s3]\"), then re-run."
            ) from exc
        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url or None,
            aws_access_key_id=access_key or None,
            aws_secret_access_key=secret_key or None,
            region_name=region,
        )
        self._bucket_ready = False
        self._lock = threading.Lock()

    def _ensure_bucket(self) -> None:
        if self._bucket_ready:
            return
        with self._lock:
            if self._bucket_ready:
                return
            from botocore.exceptions import ClientError

            try:
                self._client.head_bucket(Bucket=self._bucket)
            except ClientError:
                # Local/MinIO flow: create it. Against AWS with a
                # missing bucket this raises the service's own error,
                # which is the operator's signal to pre-create it.
                self._client.create_bucket(Bucket=self._bucket)
            self._bucket_ready = True

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        self._ensure_bucket()
        self._client.put_object(
            Bucket=self._bucket, Key=key, Body=data, ContentType=content_type
        )
        return key

    def get(self, key: str) -> bytes:
        response = self._client.get_object(Bucket=self._bucket, Key=key)
        return response["Body"].read()

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError:
            return False


def get_object_store() -> S3ObjectStore | None:
    """The configured object store, or None when no bucket is set.

    ``S3_BUCKET`` is the switch: set it (with ``S3_ENDPOINT_URL`` for
    MinIO / non-AWS endpoints) and documents resolve through object
    storage; leave it unset and documents carry their text layers
    inline — the hermetic offline path.
    """
    load_dotenv()
    bucket = env_str("S3_BUCKET")
    if not bucket:
        return None
    return S3ObjectStore(
        bucket=bucket,
        endpoint_url=env_str("S3_ENDPOINT_URL") or None,
        access_key=env_str("S3_ACCESS_KEY") or None,
        secret_key=env_str("S3_SECRET_KEY") or None,
        region=env_str("S3_REGION", "us-east-1"),
    )
