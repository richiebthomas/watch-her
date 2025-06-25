# Deployment Guide for Watch Her - Women Safety CCTV Prototype

## 🚀 Deploying to Hugging Face Spaces

### Prerequisites
- Hugging Face account (create at [huggingface.co](https://huggingface.co))
- Git installed on your machine

### Step-by-Step Deployment

#### 1. Create a New Space on Hugging Face

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Fill in the details:
   - **Space name**: `watch-her-cctv-prototype` (or your preferred name)
   - **License**: MIT
   - **SDK**: Gradio
   - **Hardware**: CPU Basic (free tier)
   - **Visibility**: Public (or Private if preferred)

#### 2. Clone the Space Repository

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/watch-her-cctv-prototype
cd watch-her-cctv-prototype
```

#### 3. Copy Your Files

Copy these files from your local project to the cloned space:

```
📁 Your Space Repository/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── .gitignore           # Git ignore file
```

**Important**: Do NOT copy the model files (*.pb, *.pbtxt, *.caffemodel) as they will be downloaded automatically.

#### 4. Commit and Push

```bash
git add .
git commit -m "Initial deployment of Watch Her CCTV prototype"
git push
```

### 🔧 Alternative: Manual File Upload

If you prefer not to use Git, you can upload files directly:

1. Go to your Space page on Hugging Face
2. Click "Files" tab
3. Click "Add file" and upload each file individually
4. Make sure to upload:
   - `app.py`
   - `requirements.txt` 
   - `README.md`

### 📋 Files Overview

#### `app.py`
- Main Gradio application
- Handles model downloading automatically
- Provides web interface for gender classification
- Processes images and displays results

#### `requirements.txt`
```
gradio>=4.0.0
opencv-python-headless>=4.8.0
numpy>=1.21.0
Pillow>=8.0.0
urllib3>=1.26.0
```

#### `README.md`
- Contains Space metadata in YAML frontmatter
- Comprehensive project documentation
- Usage instructions and disclaimers

### ⚙️ Configuration Options

#### Space Settings
You can modify these in the README.md frontmatter:

```yaml
---
title: Watch Her - Women Safety CCTV Prototype
emoji: 👁️
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---
```

#### Hardware Requirements
- **CPU Basic**: Free tier, sufficient for this application
- **CPU Upgrade**: Faster processing for multiple users
- **GPU**: Not needed for this application

### 🚦 Deployment Process

1. **Build Phase**: Hugging Face will install dependencies from `requirements.txt`
2. **Model Download**: App automatically downloads required models on first run
3. **Launch**: Gradio interface becomes available at your Space URL

### 📊 Expected Build Time
- Initial build: 3-5 minutes
- Model download: 2-3 minutes (happens on first user access)
- Total startup time: 5-8 minutes

### 🔍 Troubleshooting

#### Common Issues

**Build Fails**
- Check `requirements.txt` for typos
- Ensure all dependencies are compatible
- Check Space logs for specific errors

**Models Don't Download**
- Internet connectivity required
- Large model files may timeout on slow connections
- Check Space logs for download errors

**App Doesn't Start**
- Verify `app.py` syntax
- Check that all imports are available
- Ensure `demo.launch()` is called

#### Debugging Steps

1. **Check Space Logs**
   - Go to your Space page
   - Click "Logs" tab
   - Look for error messages

2. **Test Locally First**
   ```bash
   python app.py
   ```

3. **Verify Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### 🌐 Accessing Your Deployed App

Once deployed, your app will be available at:
```
https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME
```

### 🔒 Privacy and Security

- **Model Processing**: Happens on Hugging Face servers
- **Image Storage**: Images are not stored permanently
- **Data Privacy**: Follow Hugging Face's privacy policy
- **Usage Monitoring**: Hugging Face may monitor usage

### 📈 Monitoring and Updates

#### Viewing Usage
- Check your Space's analytics on Hugging Face
- Monitor logs for errors or issues

#### Updating Your App
1. Make changes to local files
2. Push updates via Git or upload manually
3. Space will automatically rebuild and redeploy

### 💡 Tips for Success

1. **Test Locally**: Always test your app locally before deploying
2. **Monitor Logs**: Keep an eye on Space logs after deployment
3. **User Feedback**: Consider adding feedback mechanisms
4. **Documentation**: Keep README.md updated with any changes
5. **Performance**: Monitor response times and optimize if needed

### 🚀 Going Live

After deployment:
1. Test the app thoroughly
2. Share the link with intended users
3. Monitor for any issues
4. Consider upgrading hardware if needed for more users

### 📞 Support

- **Hugging Face Community**: [discuss.huggingface.co](https://discuss.huggingface.co)
- **Documentation**: [huggingface.co/docs/hub/spaces](https://huggingface.co/docs/hub/spaces)
- **Status**: [status.huggingface.co](https://status.huggingface.co)

---

**Happy Deploying! 🚀**

Your Watch Her CCTV prototype will be live and accessible to users worldwide once deployed to Hugging Face Spaces. 