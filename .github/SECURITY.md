# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please follow these steps:

### Do NOT

- Open a public GitHub issue
- Disclose the vulnerability publicly before it is fixed
- Exploit the vulnerability

### Do

1. **Email us directly** at: nayanchandradas@hotmail.com
2. **Include details:**
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

3. **Wait for response:** We will acknowledge receipt within 48 hours

### What to Expect

1. **Acknowledgment:** Within 48 hours
2. **Initial Assessment:** Within 7 days
3. **Fix Timeline:** Depends on severity
   - Critical: 24-48 hours
   - High: 7 days
   - Medium: 30 days
   - Low: 90 days

4. **Credit:** We will credit you in the release notes (unless you prefer anonymity)

## Security Best Practices

When using FaceVerify in production:

1. **Input Validation:** Always validate image inputs before processing
2. **Rate Limiting:** Implement rate limiting for API endpoints
3. **Access Control:** Restrict access to verification endpoints
4. **Data Privacy:** Handle facial data according to privacy regulations (GDPR, etc.)
5. **Model Security:** Keep embedding models secure and up-to-date
6. **Logging:** Log verification attempts for audit purposes
7. **HTTPS:** Always use HTTPS in production

## Known Security Considerations

### Facial Data Privacy

- Face embeddings are biometric data
- Store embeddings securely (encrypted at rest)
- Implement proper access controls
- Follow data retention policies
- Obtain user consent where required

### Adversarial Attacks

- Be aware of potential spoofing attacks (photos of photos)
- Consider liveness detection for high-security applications
- Monitor for unusual verification patterns

## Security Updates

Security updates are released as patch versions (e.g., 1.0.1, 1.0.2).

Subscribe to releases to stay informed:
https://github.com/nayandas69/faceverify/releases
