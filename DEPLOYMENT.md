# 🚀 Deployment Guide for Emotion Recognition App

This guide will help you deploy your Emotion Recognition from Speech application to various platforms.

## 📋 Prerequisites

1. **Trained Model**: Make sure you have trained models in the `model/` directory
2. **Git Repository**: Your project should be in a Git repository
3. **Dependencies**: All required packages are listed in `requirements_deploy.txt`

## 🎯 Deployment Options

### Option 1: Streamlit Cloud (Recommended - Free)

**Streamlit Cloud** is the easiest way to deploy Streamlit apps for free.

#### Steps:

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Add deployment files"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set the main file path to: `streamlit_app.py`
   - Click "Deploy"

3. **Configuration**:
   - The app will automatically use `requirements_deploy.txt`
   - Your model files will be included in the deployment

### Option 2: Heroku

**Heroku** is a popular cloud platform for web applications.

#### Steps:

1. **Install Heroku CLI**:
   ```bash
   # Download from: https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Login to Heroku**:
   ```bash
   heroku login
   ```

3. **Create Heroku App**:
   ```bash
   heroku create your-app-name
   ```

4. **Deploy**:
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

5. **Open App**:
   ```bash
   heroku open
   ```

### Option 3: Railway

**Railway** is a modern deployment platform with a generous free tier.

#### Steps:

1. **Go to Railway**:
   - Visit [railway.app](https://railway.app)
   - Sign in with GitHub

2. **Deploy**:
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository
   - Railway will automatically detect it's a Python app

3. **Configure**:
   - Set the start command: `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`
   - Add environment variables if needed

### Option 4: Render

**Render** is another excellent free deployment platform.

#### Steps:

1. **Go to Render**:
   - Visit [render.com](https://render.com)
   - Sign up with GitHub

2. **Create Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Set build command: `pip install -r requirements_deploy.txt`
   - Set start command: `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`

3. **Deploy**:
   - Click "Create Web Service"
   - Wait for deployment to complete

### Option 5: Local Deployment

For testing or internal use, you can deploy locally.

#### Steps:

1. **Install Dependencies**:
   ```bash
   pip install -r requirements_deploy.txt
   ```

2. **Run the App**:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Access the App**:
   - Open your browser to `http://localhost:8501`
   - The app will be available on your local network

## 🔧 Configuration Files

### Streamlit Configuration (`.streamlit/config.toml`)
- Configures the app appearance and behavior
- Sets theme colors and server settings

### Requirements (`requirements_deploy.txt`)
- Lists all Python dependencies with pinned versions
- Ensures consistent deployment across platforms

### Procfile (for Heroku)
- Tells Heroku how to run the application
- Sets the correct port and address

### Runtime (`runtime.txt`)
- Specifies Python version for deployment platforms

## 🚨 Troubleshooting

### Common Issues:

1. **Model Not Found**:
   - Ensure model files are in the `model/` directory
   - Check file permissions
   - Verify model file names match the code

2. **Dependencies Issues**:
   - Update `requirements_deploy.txt` with correct versions
   - Some platforms may need additional system dependencies

3. **Memory Issues**:
   - Large models may exceed free tier limits
   - Consider using smaller models for deployment
   - Upgrade to paid tiers if needed

4. **Audio Processing Errors**:
   - Ensure all audio libraries are properly installed
   - Check audio file format support

### Platform-Specific Issues:

#### Streamlit Cloud:
- File size limits: 200MB per file
- Memory limits: 1GB RAM
- Build time limits: 10 minutes

#### Heroku:
- File size limits: 500MB total
- Memory limits: 512MB (free tier)
- Build time limits: 15 minutes

#### Railway:
- File size limits: 1GB
- Memory limits: 512MB (free tier)
- Build time limits: 20 minutes

## 📊 Monitoring

### Check App Status:
- Monitor deployment logs for errors
- Test audio upload functionality
- Verify model loading works correctly

### Performance Optimization:
- Use smaller models for faster loading
- Optimize audio processing
- Cache model loading with `@st.cache_resource`

## 🔒 Security Considerations

1. **File Upload**:
   - Validate audio file types
   - Limit file sizes
   - Sanitize file names

2. **Model Security**:
   - Don't expose model internals
   - Validate input data
   - Handle errors gracefully

3. **Environment Variables**:
   - Use environment variables for sensitive data
   - Don't commit API keys or secrets

## 📈 Scaling

### For High Traffic:
1. **Upgrade to Paid Tiers**:
   - More memory and CPU
   - Better performance
   - Custom domains

2. **Optimize the App**:
   - Use model quantization
   - Implement caching
   - Optimize audio processing

3. **Load Balancing**:
   - Deploy multiple instances
   - Use CDN for static assets
   - Implement rate limiting

## 🎉 Success!

Once deployed, your app will be accessible via a public URL. Users can:
- Upload audio files
- Get emotion predictions
- View confidence scores
- See model performance statistics

## 📞 Support

If you encounter issues:
1. Check the platform's documentation
2. Review deployment logs
3. Test locally first
4. Consider platform-specific forums

---

**Happy Deploying! 🚀** 