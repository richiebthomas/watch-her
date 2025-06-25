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

# Watch Her - Women Safety CCTV Prototype 🚧

## Overview

**Watch Her** is a prototype AI-powered CCTV system designed for women's safety applications. This work-in-progress project focuses on processing video streams (webcam or video files) to detect persons and classify their gender in real-time, enhancing security monitoring capabilities.

## 🎯 Current Features

- **Video Processing**: Supports both video file upload and webcam streaming
- **Person Detection**: Advanced face detection using OpenCV's DNN-based models
- **Gender Classification**: Binary gender classification (Male/Female) with confidence scores
- **Real-time Analysis**: Process video frames with visual annotations and statistics
- **Color-coded Indicators**: Green boxes for females, red boxes for males
- **Processing Statistics**: Detailed analysis summary with detection counts
- **Privacy-Focused**: Videos processed temporarily, no permanent storage
- **Web Interface**: Multi-tab Gradio interface for different input methods

## 🔧 Technology Stack

- **Computer Vision**: OpenCV with pre-trained DNN models
- **Deep Learning**: TensorFlow and Caffe model formats
- **Frontend**: Gradio for web interface
- **Backend**: Python with NumPy for image processing

## 📋 How to Use

### Video Upload Tab
1. **Upload Video**: Choose a video file (MP4, AVI, MOV, etc.)
2. **Process**: Click "Process Video" button
3. **Wait**: Processing time depends on video length (max 1000 frames for demo)
4. **Download**: Get the processed video with annotations

### Webcam Tab
1. **Record**: Use your webcam to record a video
2. **Process**: Click "Process Webcam Video" button
3. **View Results**: Get processed video with real-time annotations

### Visual Output
Each detected person will be highlighted with:
- **Green boxes**: Female detections
- **Red boxes**: Male detections
- **Labels**: Gender + confidence score
- **Statistics**: Summary of all detections

## 🎨 Model Information

### Face Detection
- **Model**: OpenCV DNN Face Detector
- **Framework**: TensorFlow
- **Input Size**: 300x300 pixels
- **Confidence Threshold**: 0.7

### Gender Classification
- **Model**: Age-Gender Deep Learning Model
- **Framework**: Caffe
- **Input Size**: 227x227 pixels
- **Classes**: Male, Female

## ⚠️ Important Notes

### Limitations
- **Prototype Status**: This is a work-in-progress and not ready for production use
- **Accuracy Variations**: Performance may vary based on:
  - Image quality and resolution
  - Lighting conditions
  - Face angle and visibility
  - Image compression

### Ethical Considerations
- **Educational Purpose**: This tool is intended for research and educational purposes only
- **Bias Awareness**: Gender classification models may have inherent biases
- **Privacy Respect**: Always ensure proper consent before using on surveillance footage
- **Responsible Use**: Consider the ethical implications of gender classification in surveillance

## 🛠️ Technical Details

### Model Files
The application automatically downloads the following pre-trained models:
- `opencv_face_detector_uint8.pb` - Face detection model
- `opencv_face_detector.pbtxt` - Face detection configuration
- `gender_deploy.prototxt` - Gender classification network architecture
- `gender_net.caffemodel` - Pre-trained gender classification weights

### Performance
- **Processing Speed**: ~5-15 FPS depending on hardware and video resolution
- **Memory Usage**: Approximately 500-1000 MB RAM during video processing
- **Supported Formats**: MP4, AVI, MOV, MKV, WebM, and other common video formats
- **Frame Limit**: Max 1000 frames per video for demo purposes

## 🚀 Future Enhancements

- [ ] Age estimation capabilities
- [ ] Emotion detection
- [ ] Multiple person tracking in video streams
- [ ] Enhanced accuracy with newer models
- [ ] Integration with actual CCTV systems
- [ ] Alert system for specific scenarios
- [ ] Mobile app development

## 📊 Example Use Cases

1. **Security Monitoring**: Real-time demographic analysis of video surveillance feeds
2. **Research**: Academic studies on computer vision and AI in security applications
3. **Education**: Learning about video processing, face detection, and classification
4. **Prototyping**: Foundation for more advanced real-time surveillance systems
5. **Event Analysis**: Post-event analysis of recorded surveillance footage

## 🤝 Contributing

This is an open-source project and contributions are welcome! Areas where help is needed:
- Model accuracy improvements
- UI/UX enhancements
- Performance optimizations
- Additional safety features
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV team for the face detection models
- Gil Levi and Tal Hassner for the age-gender classification model
- Hugging Face for the hosting platform
- Gradio team for the excellent web interface framework

## ⚖️ Disclaimer

This tool is a prototype for educational and research purposes. The developers are not responsible for any misuse of this technology. Please use responsibly and in accordance with local laws and regulations regarding surveillance and privacy.

---

**Status**: 🚧 Work in Progress  
**Version**: 0.1.0  
**Last Updated**: 2024

For issues, suggestions, or contributions, please feel free to reach out or submit a pull request. 