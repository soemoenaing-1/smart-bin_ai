# ♻️ Smart Bin AI - Waste Detection System

An intelligent waste classification system using YOLO11 computer vision and Streamlit for real-time waste material detection. This AI-powered solution automatically identifies and categorizes waste materials to promote proper recycling and waste management.

## 🚀 Features

### Core Functionality
- **Real-time Waste Detection**: Classifies waste into 5 categories using YOLO11
- **Interactive Web Interface**: Beautiful Streamlit-based UI with real-time analytics
- **Live Webcam Support**: Real-time detection through webcam streaming
- **Image Upload**: Process single images for waste classification
- **Sample Images**: Built-in sample images for testing and demonstration

### Detection Categories
- 📦 **Cardboard**: Boxes, packaging materials, paper-based containers
- 🍶 **Glass**: Bottles, jars, glass containers
- 🥫 **Metal**: Cans, aluminum foil, metal containers  
- 📄 **Paper**: Documents, newspapers, paper products
- 🧴 **Plastic**: Bottles, bags, plastic containers

### Advanced Features
- **Confidence Scoring**: Real-time confidence levels for each detection
- **Bounding Box Visualization**: Clear visual indicators around detected objects
- **Analytics Dashboard**: Detection history, statistics, and performance metrics
- **Session Tracking**: Monitor usage patterns and detection accuracy
- **Responsive Design**: Modern UI with frosted glass effects and smooth animations

## 🛠️ Technology Stack

### Machine Learning
- **YOLO11**: State-of-the-art object detection model (Ultralytics)
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision and image processing

### Web Application
- **Streamlit**: Interactive web application framework
- **Plotly**: Interactive data visualization
- **Pillow**: Image processing and manipulation

### Additional Libraries
- **NumPy**: Numerical computations
- **Pandas**: Data analysis and manipulation
- **Streamlit-WebRTC**: Real-time webcam streaming

## 📁 Project Structure

```
smart-bin/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── my_model/             # Trained YOLO model
│   ├── my_model.pt       # Pre-trained weights
│   ├── yolo_detect.py    # Standalone detection script
│   └── train/            # Training artifacts and metrics
│       ├── args.yaml     # Training configuration
│       ├── results.csv   # Training results
│       ├── results.png   # Performance graphs
│       └── weights/      # Model checkpoints
├── data/                 # Dataset structure
│   ├── classes.txt       # Class definitions
│   ├── images/           # Training images
│   └── labels/           # YOLO format annotations
├── cardboard/            # Sample cardboard images
├── glass/                # Sample glass images
├── metal/                # Sample metal images
├── paper/                # Sample paper images
├── plastic/              # Sample plastic images
├── data.zip              # Compressed dataset
└── my_model.zip          # Compressed model files
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/soemoenaing-1/smart-bin_ai.git
cd smart-bin_ai
```

2. **Create and activate virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

The application will automatically open in your web browser at `http://localhost:8501`

## 🎯 Usage Guide

### Web Interface
1. **Upload Image**: Click "Upload Image" to select a waste image for classification
2. **Use Webcam**: Click "Use Webcam" for real-time detection
3. **Sample Images**: Try built-in sample images to test the system
4. **Adjust Confidence**: Use the slider to set detection sensitivity (0.1-1.0)

### Standalone Detection
Use the command-line tool for batch processing:
```bash
python my_model/yolo_detect.py --model my_model/my_model.pt --source path/to/image.jpg --thresh 0.5
```

### Supported Input Formats
- **Images**: JPG, JPEG, PNG, BMP
- **Videos**: MP4, AVI, MOV
- **Webcam**: Real-time camera feed
- **Folders**: Batch processing of image directories

## 📊 Model Performance

### Training Details
- **Model**: YOLO11s (Small variant)
- **Dataset**: 2,392 annotated images
- **Classes**: 5 waste categories
- **Training Epochs**: 60
- **Image Size**: 640x640 pixels
- **Batch Size**: 16

### Metrics
- **mAP@0.5**: [View training results in `my_model/train/results.png`]
- **Precision**: High precision across all categories
- **Recall**: Comprehensive detection capability
- **Inference Speed**: Real-time processing capability

## 🎨 UI Features

### Design Elements
- **Modern Glass-morphism**: Frosted glass effects with backdrop filters
- **Responsive Layout**: Adapts to different screen sizes
- **Color-coded Categories**: Each waste type has distinct colors
- **Smooth Animations**: Hover effects and transitions
- **Dark/Light Theme**: Optimized for different lighting conditions

### Analytics Dashboard
- **Detection History**: Track all detections in current session
- **Category Distribution**: Pie charts showing waste type distribution
- **Confidence Trends**: Line graphs for detection confidence over time
- **Performance Metrics**: Real-time statistics and counters

## 🔧 Configuration

### Model Parameters
- **Confidence Threshold**: Adjustable (default: 0.5)
- **IoU Threshold**: 0.45 for non-maximum suppression
- **Max Detections**: 10 objects per image
- **Input Size**: 640x640 pixels

### Customization
Edit `app.py` to modify:
- Class definitions and colors
- UI styling and themes
- Detection parameters
- Analytics features

## 📚 Dataset Information

### Data Sources
- **Cardboard**: 401 images
- **Glass**: 401 images  
- **Metal**: 401 images
- **Paper**: 401 images
- **Plastic**: 501 images (including web-scraped samples)

### Annotation Format
YOLO format: `<class_id> <x_center> <y_center> <width> <height>`

### Class Mapping
```
0: Cardboard
1: Glass
2: Metal
3: Paper
4: Plastic
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Areas
- **Model Improvement**: Enhanced detection accuracy
- **UI/UX**: Better interface design
- **New Features**: Additional functionality
- **Bug Fixes**: Issue resolution
- **Documentation**: README and code comments

## 🐛 Troubleshooting

### Common Issues

**Model Loading Error**
```bash
# Ensure model file exists
ls my_model/my_model.pt
# Re-download if missing
```

**Webcam Not Working**
- Check browser permissions
- Ensure no other app is using the camera
- Try refreshing the page

**Low Detection Accuracy**
- Adjust confidence threshold
- Ensure good lighting conditions
- Use clear, high-quality images

**Memory Issues**
- Reduce batch size in training
- Use smaller image resolution
- Close unnecessary applications

### Performance Optimization
- **GPU Acceleration**: Install CUDA-compatible PyTorch
- **Model Quantization**: Reduce model size for faster inference
- **Image Resizing**: Optimize input dimensions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics**: YOLO11 framework and model architecture
- **Streamlit**: Web application framework
- **OpenCV**: Computer vision library
- **PyTorch**: Deep learning framework
- **Waste Dataset Contributors**: Various open-source waste classification datasets

## 📞 Contact

- **GitHub**: [@soemoenaing-1](https://github.com/soemoenaing-1)
- **Project Repository**: [smart-bin_ai](https://github.com/soemoenaing-1/smart-bin_ai)

## 🔮 Future Enhancements

### Planned Features
- **Multi-language Support**: Internationalization
- **Mobile App**: React Native/Flutter application
- **API Integration**: RESTful API for third-party integration
- **Cloud Deployment**: AWS/Azure/GCP deployment options
- **Advanced Analytics**: Machine learning insights
- **IoT Integration**: Smart bin hardware integration

### Research Areas
- **Waste Quantity Estimation**: Volume and weight calculation
- **Recycling Recommendations**: Proper disposal instructions
- **Location-based Services**: Nearby recycling centers
- **Gamification**: User engagement and rewards

---

**Made with ❤️ for a cleaner planet** ♻️