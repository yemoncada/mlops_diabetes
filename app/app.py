import streamlit as st
import requests
import random

label_mapping = {
    0: 'No readmitido',
    1: 'Readmitido',
}

def set_page_config():
    st.set_page_config(
        page_title="Final Deployments",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/yemoncada/mlops_diabetes',
            'Report a bug': "https://github.com/yemoncada/mlops_diabetes",
            'About': "Kubernets APP Deployment"
        }
    )

def create_header():
    st.title('Airflow Proyecto Final (CoverType Dataset)')
    st.subheader('by Yefry Moncada Linares ([@yemoncad](https://github.com/yemoncada?tab=repositories))')

    st.markdown(
        """
        <br><br/>
        La siguiente aplicación proporciona una solución completa para el procesamiento y análisis de datos utilizando
        Streamlit, Python, Docker y Kubernetes, ofreciendo una experiencia de usuario interactiva y una infraestructura robusta
        y escalable para garantizar un rendimiento óptimo en entornos productivos.

        La aplicación se centra en la implementación de servicios que permiten cargar información desde archivos de texto plano
        hacia bases de datos, entrenar modelos de inteligencia artificial, realizar inferencias con modelos previamente entrenados, 
        almacenar información utilizada en el proceso de inferencia en archivos de texto plano y ofrecer una interfaz gráfica interactiva 
        para facilitar la interacción con estos servicios.
        """
        , unsafe_allow_html=True)

    st.markdown('---')


def create_inferencia():

    # Especifique la URL de su API FastAPI
    url = 'http://10.43.102.110:8503/predict'

    # Cree las entradas para los campos de datos del paciente
    encounter_id = random.randint(1000,9999)
    patient_nbr = random.randint(1000,9999)
    race = st.selectbox( 'Raza', ('Caucasian', 'AfricanAmerican', '?', 'Other', 'Asian', 'Hispanic'))
    gender = st.selectbox( 'Gender', ('Female', 'Male', 'Unknown/Invalid'))
    age = st.selectbox('Age', ('[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)',
       '[60-70)', '[70-80)', '[80-90)', '[90-100)'))
    weight = st.selectbox('Weight', ('?', '[75-100)', '[50-75)', '[0-25)', '[100-125)', '[25-50)',
       '[125-150)', '[175-200)', '[150-175)', '>200'))
    admission_type_id = st.slider('Admission Type ID', 1, 7 )
    discharge_disposition_id = st.slider('Discharge Disposition ID', 1, 27 )
    admission_source_id = st.selectbox('Admission Source ID', (1,  7,  2,  4,  5,  6, 20,  3, 17,  8,  9, 14, 10, 22, 11, 25, 13))
    time_in_hospital = st.selectbox('Time in Hospital', (1,  3,  2,  4,  5, 13, 12,  9,  7, 10,  6, 11,  8, 14))
    payer_code = st.selectbox('Payer Code', ('?', 'MC', 'MD', 'HM', 'UN', 'BC', 'SP', 'CP', 'SI', 'DM', 'CM',
       'CH', 'PO', 'WC', 'OT', 'OG', 'MP', 'FR'))

    medical_specialty = st.selectbox('Medical Specialty', ('Pediatrics-Endocrinology', '?', 'InternalMedicine',
       'Family/GeneralPractice', 'Cardiology', 'Surgery-General',
       'Orthopedics', 'Gastroenterology',
       'Surgery-Cardiovascular/Thoracic', 'Nephrology',
       'Orthopedics-Reconstructive', 'Psychiatry', 'Emergency/Trauma',
       'Pulmonology', 'Surgery-Neuro',
       'Obsterics&Gynecology-GynecologicOnco', 'ObstetricsandGynecology',
       'Pediatrics', 'Hematology/Oncology', 'Otolaryngology',
       'Surgery-Colon&Rectal', 'Pediatrics-CriticalCare', 'Endocrinology',
       'Urology', 'Psychiatry-Child/Adolescent', 'Pediatrics-Pulmonology',
       'Neurology', 'Anesthesiology-Pediatric', 'Radiology',
       'Pediatrics-Hematology-Oncology', 'Psychology', 'Podiatry',
       'Gynecology', 'Oncology', 'Pediatrics-Neurology',
       'Surgery-Plastic', 'Surgery-Thoracic',
       'Surgery-PlasticwithinHeadandNeck', 'Ophthalmology',
       'Surgery-Pediatric', 'Pediatrics-EmergencyMedicine',
       'PhysicalMedicineandRehabilitation', 'InfectiousDiseases',
       'Anesthesiology', 'Rheumatology', 'AllergyandImmunology',
       'Surgery-Maxillofacial', 'Pediatrics-InfectiousDiseases',
       'Pediatrics-AllergyandImmunology', 'Dentistry', 'Surgeon',
       'Surgery-Vascular', 'Osteopath', 'Psychiatry-Addictive',
       'Surgery-Cardiovascular', 'PhysicianNotFound', 'Hematology',
       'Proctology', 'Obstetrics', 'SurgicalSpecialty', 'Radiologist',
       'Pathology', 'Dermatology', 'SportsMedicine', 'Speech',
       'Hospitalist', 'OutreachServices', 'Cardiology-Pediatric',
       'Perinatology', 'Neurophysiology', 'Endocrinology-Metabolism',
       'DCPTEAM', 'Resident'))
    
    num_lab_procedures = st.slider('Number of Lab Procedures', 1, 132 )
    num_procedures = st.slider('Number of Procedures', 1, 6)
    num_medications = st.slider('Number of Medications', 1, 81 )
    number_outpatient = st.slider('Number of Outpatient', 1, 42 )
    number_emergency = st.slider('Number of Emergency', 1, 76 )
    number_inpatient = st.slider('Number of Inpatient', 1, 21 )
    diag_1 = st.selectbox('diag1', ('V583','E921','NaN', '390', '459', '785'))
    diag_2 = st.selectbox('diag2', ('V583','E921','NaN', '390', '459', '785'))
    diag_3 = st.selectbox('diag3', ('V583','E921','NaN', '390', '459', '785'))
    number_diagnoses = st.slider('Number of diagnoses', 1, 16)
    max_glu_serum = st.selectbox('Max Glu Serum', ('None', '>300', 'Norm', '>200'))
    A1Cresult = st.selectbox('AC1Result', ('None', '>7', '>8', 'Norm'))
    metformin = st.selectbox('metformin ', ('No', 'Steady', 'Up', 'Down'))
    repaglinide = st.selectbox('repaglinide', ('No', 'Steady', 'Up', 'Down'))
    nateglinide = st.selectbox('nateglinide', ('No', 'Steady', 'Up', 'Down'))
    chlorpropamide = st.selectbox('chlorpropamide', ('No', 'Steady', 'Up', 'Down'))
    glimepiride = st.selectbox('glimepiride', ('No', 'Steady', 'Up', 'Down'))
    acetohexamide = st.selectbox('acetohexamide', ('No', 'Steady', 'Up', 'Down'))
    glipizide = st.selectbox('glipizide', ('No', 'Steady', 'Up', 'Down'))
    glyburide = st.selectbox('glyburide', ('No', 'Steady', 'Up', 'Down'))
    tolbutamide = st.selectbox('tolbutamide', ('No', 'Steady', 'Up', 'Down'))
    pioglitazone = st.selectbox('pioglitazone', ('No', 'Steady', 'Up', 'Down'))
    rosiglitazone = st.selectbox('rosiglitazone', ('No', 'Steady', 'Up', 'Down'))
    acarbose = st.selectbox('acarbose', ('No', 'Steady', 'Up', 'Down'))
    miglitol = st.selectbox('miglitol', ('No', 'Steady', 'Up', 'Down'))
    troglitazone = st.selectbox('troglitazone', ('No', 'Steady', 'Up', 'Down'))
    tolazamide = st.selectbox('tolazamide', ('No', 'Steady', 'Up', 'Down'))
    examide = st.selectbox('examide', ('No', 'Steady', 'Up', 'Down'))
    citoglipton = st.selectbox('citoglipton', ('No', 'Steady', 'Up', 'Down'))
    insulin = st.selectbox('insulin', ('No', 'Steady', 'Up', 'Down'))
    glyburide_metformin = st.selectbox('glyburide_metformin', ('No', 'Steady', 'Up', 'Down'))
    glipizide_metformin = st.selectbox('glipizide_metformin', ('No', 'Steady', 'Up', 'Down'))
    glimepiride_pioglitazone = st.selectbox('glimepiride_pioglitazone', ('No', 'Steady', 'Up', 'Down'))
    metformin_rosiglitazone = st.selectbox('metformin_rosiglitazone', ('No', 'Steady', 'Up', 'Down'))
    metformin_pioglitazone = st.selectbox('metformin_pioglitazone', ('No', 'Steady', 'Up', 'Down'))
    change = st.selectbox('Change', ('No', 'Ch'))
    diabetesMed = st.selectbox('Change', ('No', 'Yes'))

    # Cuando se presiona el botón, recoja los valores de las entradas y llame a la API
    if st.button('Predict'):
        with st.spinner('Ejecutando...'):
            
            data = {
                'encounter_id': encounter_id,
                'patient_nbr': patient_nbr,
                'race': race,
                'gender': gender,
                'age': age,
                'weight': weight,
                'admission_type_id': admission_type_id,
                'discharge_disposition_id': discharge_disposition_id,
                'admission_source_id': admission_source_id,
                'time_in_hospital': time_in_hospital,
                'payer_code': payer_code,
                'medical_specialty': medical_specialty,
                'num_lab_procedures': num_lab_procedures,
                'num_procedures': num_procedures,
                'num_medications': num_medications,
                'number_outpatient': number_outpatient,
                'number_emergency': number_emergency,
                'number_inpatient': number_inpatient,
                'diag_1': diag_1,
                'diag_2': diag_2,
                'diag_3': diag_3,
                'number_diagnoses': number_diagnoses,
                'max_glu_serum':max_glu_serum,
                'A1Cresult': A1Cresult,
                'metformin': metformin,
                'repaglinide': repaglinide,
                'nateglinide': nateglinide,
                'chlorpropamide': chlorpropamide,
                'glimepiride': glimepiride,
                'acetohexamide': acetohexamide,
                'glipizide': glipizide,
                'glyburide': glyburide,
                'tolbutamide': tolbutamide,
                'pioglitazone': pioglitazone,
                'rosiglitazone': rosiglitazone,
                'acarbose': acarbose,
                'miglitol': miglitol,
                'troglitazone': troglitazone,
                'tolazamide': tolazamide,
                'examide': examide,
                'citoglipton': citoglipton,
                'insulin': insulin,
                'glyburide_metformin': glyburide_metformin,
                'glipizide_metformin': glipizide_metformin,
                'glimepiride_pioglitazone': glimepiride_pioglitazone,
                'metformin_rosiglitazone': metformin_rosiglitazone,
                'metformin_pioglitazone': metformin_pioglitazone,
                'change': change,
                'diabetesMed': diabetesMed
            }

            print('[Iniciando Predicción]')

            response = requests.post(url, json=data)
            if response.status_code == 200:
                response = response.json()
                st.header(f"El paciente ha sido: {response['prediction']} - {label_mapping[response['prediction']]}")
                st.header(f"Nombre del Modelo en producción: {response['model_version']}")  
            else:
                st.write('Could not get a prediction at this time.')

def main():
    set_page_config()
    create_header()
    create_inferencia()

if __name__ == "__main__":
    main()
