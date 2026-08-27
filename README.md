# 📈 Demand Forecasting & Sales Intelligence System

An end-to-end machine learning system for retail demand forecasting using
time-series feature engineering, rolling forecasting, and XGBoost.

The project transforms historical store-level sales data into a production-
oriented forecasting pipeline and exposes the trained forecasting model
through a backend API with an interactive dashboard.

---

## 🚀 Project Overview

Accurate demand forecasting is an important component of modern retail
operations. Poor forecasts can lead to:

- Overstocking and increased inventory costs
- Stockouts and lost sales
- Inefficient workforce planning
- Poor purchasing decisions
- Reduced customer satisfaction

This project develops a machine learning based demand forecasting system
designed to predict future retail sales using historical sales patterns,
temporal features, and store-level information.

The system follows a complete ML workflow:

**Data → EDA → Feature Engineering → Model Development → Rolling Forecast →
Model Persistence → REST API → Dashboard**

---

## 🎯 Objectives

The main objectives of the project are:

1. Analyze historical retail sales data.
2. Identify temporal and store-level demand patterns.
3. Engineer features suitable for time-series forecasting.
4. Develop and evaluate machine learning forecasting models.
5. Implement a rolling/recursive forecasting strategy.
6. Persist the final trained model for inference.
7. Build a backend API for generating forecasts.
8. Provide an interactive dashboard for sales intelligence.

---

## 🧠 Machine Learning Approach

The forecasting pipeline uses an **XGBoost-based rolling forecasting model**.

Instead of treating the problem as a conventional random train-test
prediction problem, the project follows a time-aware forecasting approach.

### Key concepts

- Historical sales patterns
- Lag features
- Rolling statistics
- Temporal features
- Store-level information
- Rolling/recursive forecasting
- Time-aware model evaluation

This helps reduce temporal leakage and provides a more realistic estimate
of how the model would behave when forecasting future observations.

---

## 🏗️ System Architecture

```text
                    Historical Sales Data
                            │
                            ▼
                    ┌───────────────┐
                    │   EDA         │
                    │ 01_EDA.ipynb  │
                    └───────┬───────┘
                            │
                            ▼
                  ┌─────────────────────┐
                  │ Feature Engineering │
                  │ 02_Feature_         │
                  │ Engineering.ipynb   │
                  └──────────┬──────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ Model Training │
                    │ 03_Modelling   │
                    └───────┬────────┘
                            │
                            ▼
                  ┌─────────────────────┐
                  │ Final Model         │
                  │ 04_FinalModelling   │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Trained XGBoost     │
                  │ Model + Features    │
                  └──────────┬──────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │   FastAPI      │
                    │   Backend      │
                    └───────┬────────┘
                            │
                            ▼
                    ┌────────────────┐
                    │   Dashboard    │
                    │   Frontend     │
                    └────────────────┘
