"""S3 (or MinIO) object storage wrapper for raw uploads + raw voice audio."""
from __future__ import annotations

import boto3
from botocore.client import Config

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)


def _client():
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url or None,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name=settings.s3_region,
        config=Config(signature_version="s3v4"),
    )


def ensure_bucket() -> None:
    s3 = _client()
    try:
        s3.head_bucket(Bucket=settings.s3_bucket)
    except Exception:  # noqa: BLE001
        log.info("s3.create_bucket", bucket=settings.s3_bucket)
        try:
            s3.create_bucket(Bucket=settings.s3_bucket)
        except Exception:  # noqa: BLE001
            log.exception("s3.create_bucket_failed", bucket=settings.s3_bucket)


def put_bytes(key: str, data: bytes, content_type: str) -> str:
    s3 = _client()
    s3.put_object(
        Bucket=settings.s3_bucket, Key=key, Body=data, ContentType=content_type
    )
    log.info("s3.put", key=key, size=len(data))
    return key


def get_bytes(key: str) -> bytes:
    s3 = _client()
    obj = s3.get_object(Bucket=settings.s3_bucket, Key=key)
    return obj["Body"].read()


def delete_object(key: str) -> None:
    s3 = _client()
    s3.delete_object(Bucket=settings.s3_bucket, Key=key)
    log.info("s3.delete", key=key)
