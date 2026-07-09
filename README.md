
🎮 Video Game Sales Prediction System
A production-ready machine learning system for predicting whether video games will go on sale next, built with classification models, comprehensive infrastructure, and an interactive web dashboard.

Problem: Given historical video game data (5.47M records, 32 features), predict if a game will be discounted/on sale in the next observation period.

Solution: Binary classification using Logistic Regression, Random Forest, and Gradient Boosting models, deployed on AWS with monitoring, auto-scaling, and a real-time prediction dashboard.

🎯 Key Features
Classification Models: 3 optimized classifiers with 6 comprehensive metrics (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
Production Infrastructure: Full AWS Terraform setup (ECS Fargate, ALB, auto-scaling, monitoring, persistent storage)
Interactive Dashboard: Web-based prediction interface with model selection and real-time results
REST API: Complete Flask API with health checks, batch predictions, and model information endpoints
Comprehensive Documentation: 7 guides covering setup, API usage, changes, and deployment
📊 Project Structure
Video_Games_ML_Project/
├── data/                           # Dataset at different stages
│   ├── raw/                        # Original 5.47M record dataset
│   ├── interim/                    # Intermediate processed data
│   └── processed/                  # Final feature-engineered data
├── src/                            # Modularized source code
│   ├── data_ingestion.py          # Load and validate raw data
│   ├── features.py                # Feature engineering (10 features, stratified split, class weights)
│   ├── train.py                   # Train 3 classifiers with metrics
│   └── predict.py                 # Batch prediction interface
├── models/                         # Trained model artifacts
│   ├── logistic_regression.joblib  # Generated during training
│   ├── random_forest.joblib        # Generated during training
│   └── gradient_boosting.joblib    # Generated during training
├── reports/                        # Generated predictions and metrics
│   ├── metrics_summary.csv         # Training metrics
│   └── predictions.csv             # Batch prediction results
├── infra/                          # Production infrastructure
│   ├── main.tf                     # AWS ECS, ALB, storage, monitoring
│   ├── security.tf                 # IAM roles and security groups
│   ├── variables.tf                # Terraform configuration
│   ├── outputs.tf                  # Deployment outputs
│   ├── Dockerfile                  # Container image
│   ├── terraform.tfvars.example    # Configuration template
│   └── website/                    # Web dashboard
│       ├── index.html              # Interactive UI (500+ lines)
│       └── script.js               # API client (300+ lines)
├── tests/                          # Unit tests
├── app.py                          # Flask API (437 lines)
├── requirements.txt                # Python dependencies
├── docker-compose.yml              # Local development environment
└── README.md                        # This file
🔍 Dataset
Source: Video Game Market Price and Revenue Dataset (Kaggle)

Size: 5,470,635 records, 32 features

Target Variable: target_is_on_sale_next_obs (binary, 16.6% positive class)

Key Features Used:

discount_pct - Discount percentage
is_on_sale - Current sale status
discount_frequency_proxy - Historical discount frequency
discount_intensity - Average discount magnitude
price_vs_launch - Current price vs launch price
And 5 additional engineered features
⚡ Quick Start
1. Install Dependencies
cd Video_Games_ML_Project
pip install -r requirements.txt
2. Train the Models (15-20 minutes)
python src/data_ingestion.py      # Load and validate data
python src/features.py            # Engineer features
python src/train.py               # Train 3 classifiers
Output:

3 trained models in models/
Training metrics in reports/metrics_summary.csv
3. Start the API
python app.py
API runs on: http://localhost:5000

Health check: curl http://localhost:5000/health

4. Access the Dashboard
Open in browser: http://localhost:5000/infra/website/index.html

Features:

Model selector (pick any of 3 models)
Real-time prediction form
Probability visualization
API status monitoring
🚀 API Endpoints
All endpoints return JSON. See API_DOCUMENTATION.md for detailed specifications.

Core Endpoints
Endpoint	Method	Purpose
/health	GET	Health check & API status
/models	GET	List available models
/predict	POST	Single prediction with probability
/predict-batch	POST	Batch predictions (up to 1000 records)
/info	GET	Model metadata & training metrics
Example: Single Prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "random_forest",
    "features": {
      "discount_pct": 15.5,
      "is_on_sale": 1,
      "discount_frequency_proxy": 8,
      "discount_intensity": 18.2,
      "price_vs_launch": 0.85,
      "feature_6": 100,
      "feature_7": 50,
      "feature_8": 25,
      "feature_9": 200,
      "feature_10": 12
    }
  }'
Response:

{
  "prediction": "Will Be On Sale",
  "probability": 0.87,
  "model": "random_forest",
  "confidence": "High"
}
📈 Model Performance
Models trained on 80% of data, evaluated on 20% holdout set:

Model	Accuracy	Precision	Recall	F1-Score	ROC-AUC
Logistic Regression	TBD*	TBD*	TBD*	TBD*	TBD*
Random Forest	TBD*	TBD*	TBD*	TBD*	TBD*
Gradient Boosting	TBD*	TBD*	TBD*	TBD*	TBD*
*Metrics generated after first training run. Check reports/metrics_summary.csv

🐳 Docker Deployment
Local Development
docker-compose up
# Runs Flask API + optional services
AWS Production
cd infra/
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your AWS configuration
terraform init
terraform plan
terraform apply
AWS Stack includes:

ECS Fargate (serverless containers)
Application Load Balancer with health checks
Auto-scaling (CPU 70%, Memory 80%)
EFS (persistent storage for models)
S3 (model versioning)
DynamoDB (metadata)
CloudWatch (centralized logging & dashboards)
📚 Documentation
Document	Purpose
QUICK_START.md	Step-by-step setup guide
API_DOCUMENTATION.md	Complete API reference
API_CHANGES.md	Migration guide (regression → classification)
CHANGES_SUMMARY.md	Detailed changelog by component
COMPLETION_REPORT.md	Project transformation summary
CLEANUP_REPORT.md	Folder cleanup details
🔧 Development
Run Tests
pytest tests/
Linting
pylint src/ app.py
Feature Engineering
Edit src/features.py to modify:

Feature selection strategy
Number of features (k parameter)
Train/test split ratio
Class weight handling
Model Configuration
Edit src/train.py to modify:

Model hyperparameters
Training/evaluation metrics
Class balance strategies
🛠️ Architecture Decision Log
Why Binary Classification?

Original regression problem (price decay prediction) had low business value
Classification problem (sale prediction) is directly actionable
16.6% positive class provides realistic imbalance for real-world scenarios
Why 3 Models?

Logistic Regression: Fast, interpretable, baseline
Random Forest: Handles non-linearity, robust
Gradient Boosting: Maximum performance, recommended for production
Why AWS Fargate?

No EC2 management overhead
Automatic scaling based on demand
Pay-per-use pricing
Integrates with ALB for load balancing
Why EFS + S3 Storage?

EFS: Fast model access across tasks
S3: Version control and backup for artifacts
📝 Troubleshooting
Models not loading?
Ensure training completed: check models/ folder for .joblib files
Restart Flask app: python app.py
Dashboard showing "API Offline"?
Check Flask is running: curl http://localhost:5000/health
Check browser console for CORS errors
Try refreshing page
Batch predictions failing?
Verify CSV format matches feature names
Check feature values are numeric
Limit batch size to 1000 records
Terraform deployment failing?
Verify AWS credentials: aws sts get-caller-identity
Check VPC/subnet IDs in terraform.tfvars
Review CloudWatch logs: aws logs tail /ecs/video-game-predictor
📊 Problem Comparison
Aspect	Original (Regression)	Current (Classification) ⭐
Problem	Predict price decay	Predict if on sale next
Target	Continuous (0-100)	Binary (Yes/No)
Models	Linear, Ridge, Lasso	Logistic, RF, Gradient Boosting
Output	Predicted price	Probability + label
Business Value	Low	High
Actionability	Difficult	Direct pricing decisions
Infrastructure	None	Full production AWS stack
📄 License
[Add your license here]

👤 Author
Pablo H. F. - Data Scientist

Contact: [Add contact info]

🙏 Acknowledgments
Dataset: Kaggle - Video Game Market Price Dataset
Framework: scikit-learn, Flask, Terraform
Infrastructure: AWS (ECS, ALB, CloudWatch)
📞 Support
For issues, questions, or contributions:

Check QUICK_START.md for setup help
Review API_DOCUMENTATION.md for API questions
Check Troubleshooting section above
Open an issue in the repository
Last Updated: July 9, 2026 | Status: ✅ Prod
