from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Define possible intents and responses
intents = {
    "courses": "We offer courses in Computer Science, Mechanical, Civil, and Electronics Engineering.",
    "admission": "Admissions are open from May to July every year.",
    "fees": "The fee structure depends on the course. Contact our office for more details.",
    # We added content
    "eligibility": "Generally, you need to have passed your 10th standard (SSLC/THSLC or equivalent) with a minimum percentage of marks (usually 35%). Some courses may have specific subject requirements (like Mathematics, Science, and English).",
    "application_process": "The application process usually involves filling out an online application form on the college website or through a common admission portal for polytechnics in Kerala. You'll need to upload scanned copies of your documents and pay the application fee online. Some colleges might also have an offline application process.",
    "application_deadline": "The last date to apply for admissions varies each year. You'll need to check the official college website or the Directorate of Technical Education (DTE) Kerala website for the exact dates for the current academic year.",
    "entrance_exams": "Generally, admissions to polytechnic colleges in Kerala are based on merit (your marks in the qualifying exam). There might not be a separate entrance exam, but it's best to confirm this on the college website or by contacting them.",
    "required_documents": "Typically, you'll need the following documents: 10th standard mark sheet, Transfer Certificate (TC), Conduct Certificate, Identity proof (Aadhaar card, etc.), Caste certificate (if applicable), Passport-size photographs.",
    "reservation": "Yes, there are reservation policies in place for admissions to polytechnic colleges in Kerala, as per the government norms. These reservations usually include seats for: Scheduled Castes (SC), Scheduled Tribes (ST), Other Backward Communities (OBC), Persons with Disabilities (PWD).",
    
    "courses_offered": "Government Polytechnic College, Perumbavoor offers the following diploma programs: Diploma in Computer Engineering, Diploma in Electronics and Communication Engineering, Diploma in Mechanical Engineering, and D.Voc in Graphics and Multimedia. Each of these programs has an intake capacity of 60 students.",
    "diploma_certificate_programs": "Yes, the college offers diploma programs in Computer Engineering, Electronics and Communication Engineering, and Mechanical Engineering. Additionally, there is a D.Voc (Diploma of Vocation) program in Graphics and Multimedia.",
    "course_duration": "The diploma programs typically have a duration of three years. The D.Voc program's duration is also generally three years, but it's advisable to confirm the exact duration by contacting the college directly.",
    "subjects_covered": "For detailed information about the subjects covered in a specific course, such as the Diploma in Computer Engineering, it's best to consult the official curriculum provided by the college or the Department of Technical Education, Kerala. You can visit the Department's website for more information: https://www.sbte.kerala.gov.in/ins/01-00037.",
    "part_time_evening_courses": "Based on the available information, Government Polytechnic College, Perumbavoor primarily offers full-time diploma programs. There is no indication of part-time or evening courses being available. For the most accurate and up-to-date information, it's recommended to contact the college administration directly.",
    
    "fee_structure": "The annual fee for the Diploma in Mechanical Engineering is approximately ₹2,580 per year. For other courses, specific fee details are not readily available. It's recommended to contact the college administration directly for the most accurate and up-to-date fee information.",
    "scholarships_financial_aid": "Information regarding scholarships or financial aid options at Government Polytechnic College, Perumbavoor is not explicitly available in the provided sources. However, many government polytechnic colleges in Kerala offer scholarships based on merit and need. It's advisable to reach out to the college administration or visit the official website for detailed information on available scholarships and eligibility criteria.",
    "installment_payment_options": "There is no specific information available about installment payment options for fees at this college. To obtain accurate details, please contact the college administration directly.",
    "refund_policy": "Details regarding the refund policy for fees are not provided in the available sources. For comprehensive information on the refund policy, it's best to consult the college administration.",
    
    "campus_facilities": "The college offers a range of facilities to support student learning and well-being, including:\n\n- **Laboratories:** Well-equipped labs for various departments to facilitate practical learning.\n- **Library:** A resourceful library housing 2,858 titles and 8,518 volumes, along with subscriptions to 9 national and 2 international journals.\n- **Hostel:** Accommodation facilities are available for students.\n- **Sports Complex:** Facilities to promote physical activities and sports.\n- **Cafeteria:** Provides food services for students and staff.\n- **Auditorium:** A space for events and seminars.\n- **Gym:** Fitness facilities for students.\n- **Medical Facilities:** On-campus medical support for emergencies.\n- **Wi-Fi Campus:** Internet connectivity across the campus.",
    "hostel_facilities": "Yes, the college provides hostel facilities for students. Specific details about the charges are not readily available. It's recommended to contact the college administration directly for the most accurate and up-to-date information regarding hostel fees.",
    "cafeteria_mess_service": "Yes, there is a cafeteria on campus that offers food services to students and staff.",
    "extracurricular_activities": "The college encourages students to engage in various extracurricular activities to foster holistic development. While specific clubs are not listed, students have opportunities to participate in activities such as dance, singing, art, literature, anchoring, event management, modeling, and drama.",
    "sports_facilities": "Yes, the campus includes a sports complex that provides facilities for various sports and physical activities, promoting a healthy lifestyle among students.",
    
    "placement_assistance": "Yes, Government Polytechnic College, Perumbavoor offers placement assistance to its students. The college has a dedicated placement cell that facilitates recruitment opportunities for students.",
    "recruiting_companies": "Specific details about the companies that visit the campus for recruitment are not readily available. For the most accurate and up-to-date information, it's recommended to contact the college's placement cell directly.",
    "placement_record": "Detailed statistics regarding the college's placement record, such as placement percentages or average salary packages, are not publicly disclosed. For comprehensive information, please reach out to the college administration or placement cell.",
    "internships": "The college participates in the 'Industry on Campus' initiative, a joint program by ASAP Kerala and the Technical Education Department. This initiative aims to provide students with practical exposure and opportunities to engage in production-related activities during their course of study.",
    "industry_tie_ups": "Through the 'Industry on Campus' program, the college collaborates with various industries to enhance students' practical skills and employability. This initiative fosters a culture of 'Earn while Learn' by setting up micro-production units and providing hands-on training with state-of-the-art machinery.",
}

# Preprocess data for vectorization
intent_phrases = [
    "What courses do you offer?",
    "Tell me about admissions.",
    "What is the fee structure?",
    # We added content
    "What are the eligibility criteria for admission?",
    "What is the application process for the polytechnic courses?",
    "What is the last date to apply for admissions this year?",
    "Are there entrance exams for admission? If yes, what is the syllabus?",
    "What documents are required during the admission process?",
    "Is there any reservation or quota system for admissions?",
    
    "What courses or programs are offered at this polytechnic college?",
    "Are there any diploma or certificate programs available?",
    "What is the duration of the courses offered?",
    "What are the subjects covered in [specific course]?",
    "Does the college offer part-time or evening courses?",
    
    "What is the fee structure for various courses?",
    "Are there any scholarships or financial aid options available?",
    "Do you offer installment payment options for fees?",
    "Is there any refund policy for fees?",
    
    "What facilities are available on campus (labs, libraries, hostels)?",
    "Do you provide hostel facilities? What are the charges?",
    "Is there a cafeteria or mess service available for students?",
    "What extracurricular activities or clubs are available for students?",
    "Are there sports facilities on campus?",
    
    "Does the college offer placement assistance after course completion?",
    "Which companies visit the campus for recruitment?",
    "What is the placement record of the college?",
    "Are internships provided during the course?",
    "Does the college have tie-ups with any industries for training?",
]

responses = list(intents.values())

# Vectorize intents
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(intent_phrases)

# Simulated NLP chatbot response function
def chatbot_response(user_input):
    user_vector = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(user_vector, X)

    # Get the best match based on cosine similarity
    best_match_index = np.argmax(similarity_scores)
    best_match_score = similarity_scores[0][best_match_index]

    # Set a similarity threshold to avoid irrelevant matches
    if best_match_score > 0.3:  # Threshold can be adjusted
        return responses[best_match_index]
    else:
        return "I'm sorry, I don't understand. Can you ask something else?"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.json.get("message")
    response = chatbot_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
