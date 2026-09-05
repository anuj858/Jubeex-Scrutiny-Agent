from extraction_review.callbacks import sign_callback


def test_callback_signature_is_stable() -> None:
    body = b'{"agent_job_id":"abc"}'
    first = sign_callback(body, secret="secret", timestamp="100")
    second = sign_callback(body, secret="secret", timestamp="100")
    assert first == second
    assert first != sign_callback(body, secret="other", timestamp="100")
