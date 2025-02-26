#AiSystem - Secure Forensic Image Classification and Management System

## Overview

Project1 is a Flask-based web application designed for forensic image classification and case management. This project is built with a strong focus on security and privacy. It incorporates various measures to protect sensitive data and prevent abuse.

## Security Features

- **Sensitive Data Management:**  
  All sensitive configurations (database credentials, mail server settings, etc.) are stored in an environment file (`.env`) which is excluded from version control.

- **Rate Limiting:**  
  Critical endpoints (e.g., login and registration) are protected using Flask-Limiter to prevent brute-force attacks.

- **Email Notifications & Throttling:**  
  The system sends email notifications via Flask-Mail when adversarial attacks are detected—only if 30 minutes have passed since the last alert.

- **Activity Logging & Audit Trails:**  
  Key actions (e.g., login, registration, profile updates, case creation, image uploads, edits, and deletions) are logged in an `activity_logs` collection, available for admin review.

- **Custom Error Handling:**  
  Friendly error pages for 404 and 500 errors ensure that internal details are not exposed.


## Features

- **Authentication & Role-Based Access:**  
  - Secure registration and login.
  - Separate dashboards for regular users and admins.

- **Case Management:**  
  - Create, edit, and soft-delete cases.
  - View detailed case pages with uploaded images.

- **Image Upload & Classification:**  
  - Upload images to cases.
  - Automatically classify images using pretrained ML models.
  - Soft-delete images with distinct handling for file paths and GridFS.

- **Editing Capabilities:**  
  - Update case details (title, description).
  - Edit image comments.

- **Analytics:**  
  - View overall image classification analytics using a pie chart.
  - Analyze metrics per case.

- **Activity Logs / Audit Trails:**  
  - Detailed logs for key actions are stored for admin review.

- **API Endpoints:**  
  - RESTful endpoints for retrieving cases, uploading images, updating profiles, etc.

## Setup & Installation

1. **Create a Virtual Environment and Install Dependencies:** 

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   
2. Configure Environment Variables
   
3. Run the Application
   python app.py

    

