import streamlit as st
import cv2
import smtplib
import pandas as pd
from email.message import EmailMessage
from tensorflow.keras.models import load_model
import numpy as np
import os

# Streamlit Page Config
st.set_page_config(page_title="Stress Detection", layout="wide")

# Title and Styling
st.title("🧠 Real-Time Student Stress Detection")
st.write("Detect stress using facial expressions & questionnaire insights.")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Student Details", "Stress Detection", "Student Stress Questionnaire"])

# Load Model
model = load_model("model/stress_detection_model.h5")
input_shape = model.input_shape[1:3]  # (height, width)
channels = model.input_shape[-1]  # 1 (Grayscale) or 3 (RGB)
categories = ["Stressed", "Neutral", "Relaxed"]

# Ask for Student Details
if page == "Student Details":
    st.subheader("📝 Enter Student Details")
    name = st.text_input("Student Name")
    uid = st.text_input("University ID")
    section = st.text_input("Section")

    if st.button("Save Details"):
        if name and uid and section:
            st.session_state["name"] = name
            st.session_state["uid"] = uid
            st.session_state["section"] = section
            st.success("✅ Student details saved successfully!")
        else:
            st.error("⚠️ Please enter all details.")

# Email Notification Function
def send_email(name, uid, section, stress_score):
    email_sender = "mohitcode001@gmail.com"
    email_password = "zsct oswe ipkp zmjd"
    email_receiver = "mohitsh04321@gmail.com"
    
    subject = " Urgent: High Stress Detected in Student"
    body = (
        f" High stress detected!\n\n"
        f" Student Name: {name}\n"
        f" University ID: {uid}\n"
        f" Section: {section}\n"
        f" Stress Score: {stress_score}/100\n\n"
        " Immediate attention may be needed!"
    )
    
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = email_sender
    msg["To"] = email_receiver
    
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(email_sender, email_password)
        server.send_message(msg)
        server.quit()
        st.success("📧 Alert sent to the university counselor.")
    except Exception as e:
        st.error(f"Failed to send email: {e}")

# CSV File Storage
# CSV File Storage
def save_to_csv(data):
    file_path = "stress_data.csv"
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=["Name", "UID", "Section", "Stress Source", "Stress Level"])
        df.to_csv(file_path, index=False)
    
    df = pd.read_csv(file_path)
    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)  # Use concat instead of append
    df.to_csv(file_path, index=False)


# ---- PAGE 1: Stress Detection ----
if page == "Stress Detection":
    if "name" not in st.session_state:
        st.error("⚠️ Please enter student details first!")
    else:
        st.subheader("📷 Live Webcam Stress Detection")
        
        start = st.button("Start Webcam")
        stop = st.button("Stop Webcam", key="stop")

        frame_window = st.empty()
        stress_text = st.empty()
        progress_bar = st.progress(0)

        if start:
            cap = cv2.VideoCapture(0)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame. Please check your webcam.")
                    break

                if channels == 1:
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    face = cv2.resize(frame_gray, input_shape) / 255.0
                    face = np.reshape(face, (1, input_shape[0], input_shape[1], 1))
                    frame_display = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
                else:
                    face = cv2.resize(frame, input_shape) / 255.0
                    face = np.reshape(face, (1, input_shape[0], input_shape[1], 3))
                    frame_display = frame

                prediction = model.predict(face)[0]
                stress_label = categories[np.argmax(prediction)]
                stress_prob = np.max(prediction) * 100

                frame_window.image(frame_display, caption=f"🧠 Stress Level: {stress_label} ({stress_prob:.2f}%)", channels="BGR")
                stress_text.write(f"Predicted Stress Level: **{stress_label}** ({stress_prob:.2f}%)")
                progress_bar.progress(int(stress_prob))
                
                if stress_prob >= 80:
                    send_email(st.session_state["name"], st.session_state["uid"], st.session_state["section"], stress_prob)
                    save_to_csv({
                        "Name": st.session_state["name"],
                        "UID": st.session_state["uid"],
                        "Section": st.session_state["section"],
                        "Stress Source": "Face Detection",
                        "Stress Level": stress_prob
                    })

                if stop:
                    cap.release()
                    break
            
            cap.release()

# ---- PAGE 2: Stress Questionnaire ----
elif page == "Student Stress Questionnaire":
    if "name" not in st.session_state:
        st.error("⚠️ Please enter student details first!")
    else:
        st.subheader("📜 20 Questions to Detect Student Stress")

        questions = [
            "Do you often feel overwhelmed with assignments?",
            "Do you experience difficulty concentrating in class?",
            "Do you frequently feel anxious about exams?",
            "Have you noticed a decline in your sleep quality?",
            "Do you often skip meals due to stress?",
            "Do you experience headaches or muscle tension?",
            "Do you find it hard to balance academic and personal life?",
            "Do you feel emotionally exhausted after studying?",
            "Do you often procrastinate due to stress?",
            "Do you avoid social interactions because of academic pressure?",
            "Do you feel unmotivated to attend lectures?",
            "Do you experience mood swings or irritability?",
            "Do you struggle to manage your time effectively?",
            "Do you feel isolated from friends and family due to academic work?",
            "Do you feel pressure to perform exceptionally well?",
            "Do you feel a lack of support from professors or peers?",
            "Do you feel like you are constantly racing against deadlines?",
            "Do you experience negative thoughts about your academic abilities?",
            "Do you feel burnt out even before exams start?",
            "Do you find it hard to relax even in your free time?"
        ]

        responses = [st.radio(f"Q{i+1}: {q}", ["No", "Sometimes", "Yes"], index=1) for i, q in enumerate(questions)]

        if st.button("Analyze Stress Level"):
            stress_score = responses.count("Yes") * 5 + responses.count("Sometimes") * 2
            st.subheader("📊 Stress Level Analysis")

            if stress_score >= 80:
                st.error("⚠️ High Stress Level! Consider seeking professional help or relaxation techniques.")
                send_email(st.session_state["name"], st.session_state["uid"], st.session_state["section"], stress_score)
            
            save_to_csv({
                "Name": st.session_state["name"],
                "UID": st.session_state["uid"],
                "Section": st.session_state["section"],
                "Stress Source": "Questionnaire",
                "Stress Level": stress_score
            })

            st.write(f"**Total Stress Score:** {stress_score} / 100")
