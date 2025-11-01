# 🤗 HuggingFace Configuration Guide

DeepSynth now includes a user-friendly interface for managing your HuggingFace credentials directly from the web UI.

## ✨ Features

### 🔧 Easy Configuration
- **Visual Interface**: Configure HF token and username directly in the web UI
- **Auto-Detection**: Automatically loads existing configuration from `.env` file
- **Real-time Validation**: Validates token format and tests connection
- **Secure Storage**: Saves credentials to `.env` file for persistence

### 🛡️ Security
- **Token Masking**: Displays masked tokens in the UI for security
- **Format Validation**: Ensures tokens start with `hf_` prefix
- **Local Storage**: Credentials stored locally in `.env` file, not in browser

## 🚀 How to Use

### 1. Access Configuration
When you open the DeepSynth web interface, you'll see the HuggingFace configuration section at the top:

```
🤗 HuggingFace Configuration    [✅ Configured / ⚠️ Not configured]
```

### 2. Get Your HuggingFace Token
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token with **Write** permissions
3. Copy the token (starts with `hf_`)

### 3. Configure in DeepSynth
1. **Paste your token** in the "HuggingFace Token" field
2. **Enter your username** (your HuggingFace username)
3. **Click "Save Configuration"**
4. **Test connection** with the "Test Connection" button

### 4. Verification
- ✅ **Green status**: Configuration is working
- ⚠️ **Yellow status**: Configuration incomplete
- ❌ **Red status**: Configuration error

## 🔄 How It Works

### Backend Process
1. **Load**: Reads existing `HF_TOKEN` and `HF_USERNAME` from environment
2. **Validate**: Checks token format and requirements
3. **Save**: Updates `.env` file with new credentials
4. **Apply**: Updates runtime environment variables

### File Updates
When you save configuration, DeepSynth updates your `.env` file:

```bash
# Before
HF_TOKEN=
HF_USERNAME=

# After
HF_TOKEN=hf_your_actual_token_here
HF_USERNAME=your-username
```

## 🔧 API Endpoints

### GET `/api/config/hf-token`
Returns current configuration status:

```json
{
  "hf_token": "hf_actual_token",
  "hf_token_masked": "hf_vAT...tEs",
  "hf_username": "baconnier",
  "token_configured": true
}
```

### POST `/api/config/hf-token`
Saves new configuration:

```json
{
  "hf_token": "hf_your_new_token",
  "hf_username": "your-username"
}
```

## 🛠️ Troubleshooting

### Common Issues

**❌ "Invalid token format"**
- Solution: Ensure token starts with `hf_`
- Get a new token from HuggingFace settings

**❌ "Connection failed"**
- Solution: Check token permissions (needs Write access)
- Verify username is correct

**❌ "Token is required"**
- Solution: Enter a valid HuggingFace token
- Don't leave the field empty

### Manual Configuration
If the UI doesn't work, you can still configure manually:

```bash
# Edit .env file directly
nano .env

# Add these lines:
HF_TOKEN=hf_your_token_here
HF_USERNAME=your-username
```

## 🎯 Benefits

### Before (Manual Configuration)
```bash
# User had to:
1. Edit .env file manually
2. Remember exact variable names
3. Restart application
4. No validation or testing
```

### After (UI Configuration)
```bash
# User can:
1. Configure visually in web interface
2. Get real-time validation
3. Test connection immediately
4. See configuration status
```

## 🔐 Security Notes

- **Local Storage**: Tokens are stored in your local `.env` file
- **No Cloud Storage**: Credentials never leave your machine
- **Masked Display**: UI shows masked tokens for security
- **Validation**: Format and connection validation before saving

---

*This feature makes DeepSynth more user-friendly while maintaining security best practices.*