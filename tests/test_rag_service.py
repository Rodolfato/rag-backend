from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

expected_answer = "Sí Neurosky MindWave se utiliza como sensor de ondas cerebrales EEG"
system_answer_vectorial = "No tengo información suficiente para responder a la pregunta específica. La información disponible es sobre el dispositivo MindWave y su conexión con TGC (ThinkGear Connector) y MO (Multimodal Observer), pero no hay detalles sobre cómo se utiliza NeuroSky MindWave en MultimodalObserver."
system_answer_hybrid = "Sí, se utiliza NeuroSky MindWave en Multimodal Observer (MO). Según el contexto proporcionado, MO es una herramienta diseñada para capturar y visualizar datos multimodales, y se implementan extensiones para la captura de datos desde dispositivos externos como el EEG MindWave y el eye tracker The Eye Tribe. La extensión US-12 Capturar datos desde el sensor Mind Wave menciona explícitamente que se conecta a la API del sensor MindWave para almacenar los datos de las ondas cerebrales captadas. Esto sugiere que MO utiliza NeuroSky MindWave como dispositivo de captura de datos EEG. Además, la tabla 3.10 muestra que el prototipo P07 tiene una descripción que incluye la visualización de la posición del mouse y la captura desde dispositivos EEG MindWave y eye tracker The Eye Tribe. Esto también sugiere que MO se utiliza con NeuroSky MindWave para esta función. En resumen, según el contexto proporcionado, se utiliza NeuroSky MindWave en Multimodal Observer (MO) para capturar datos de EEG y conectarlos a la visualización del mouse y otros dispositivos externos."


expected_answers = [
    "APIs o SDKs para BITalino para la captura de señales del cuerpo",
    "Para lograr esto se utiliza un enfoque metodológico inspirado en el framework de desarrollo llamado Discipline Agile Delivery (DAD) teniendo como stakeholder o interesado principal al profesor guía del proyecto de titulación. DAD se define como un framework de procesos para el desarrollo incremental iterativo de soluciones en el ámbito de tecnologías de la información y que reúne prácticas y estrategias de otras metodologías y frameworks de desarrollo como: Scrum, XP, Agile Modeling, Unified Process, Agile Data y Kanban. Se divide en tres fases consecutivas concepción, construcción y transición.",
    "Sensor de ondas cerebrales EEG Neurosky MindWave y rastreador ocular The Eye Tribe",
    "Sí Neurosky MindWave se utiliza como sensor de ondas cerebrales EEG",
    "Se decide usar un acercamiento metodológico inspirado en Rapid Application Development (RAD) y Extreme Programming (XP). Por otra parte, toda la gestión de tareas se inspira en la metodología Kanban.",
]

system_answers_vectorial = [
    "No tengo información sobre el uso del Bitalino en Multimodal Observer. La información proporcionada solo menciona el desarrollo de Multimodal Observer como software para la captura y visualización de datos multimodales, y no hace referencia a su uso con dispositivos específicos como el Bitalino",
    "La información relevante para responder a tu pregunta es la siguiente: Según el texto proporcionado, no se especifica explícitamente la metodología utilizada en Multimodal Observer (MO). Sin embargo, se menciona que MO pretende reunir todas las funcionalidades requeridas por los investigadores en un único entorno, lo que sugiere que puede utilizar diferentes herramientas y técnicas para captura y visualización de datos multimodales. En el texto, también se menciona que MO es una herramienta arquitecturalmente modular y extensible que permite la incorporación de nuevas extensiones creadas por desarrolladores contribuidores. Esto sugiere que la metodología utilizada en MO puede variar dependiendo del desarrollo y la implementación por parte de los contribuyentes. En general, no se proporcionan detalles sobre las metodologías específicas utilizadas en MO para captura y visualización de datos multimodales. Si necesitas más información, te recomendaría buscar fuentes adicionales o consultar directamente con el autor del software.",
    "Según el contexto proporcionado, no se menciona explícitamente qué sensores específicos se utilizaron en Multimodal Observer. Sin embargo, se hace referencia a la implementación de herramientas como Eye-tracking y análisis de datos EEG. En la Tabla 3.9, se menciona que durante la iteración del sexto prototipo funcional, se integra la captura de eventos de mouse y del teclado. Esto sugiere que los sensores de entrada pueden incluir hardware de teclado y ratón (mouse), pero no se proporciona información sobre otros sensores. En la Tabla 3.11, se menciona que en el octavo prototipo funcional, se desarrolló una visualización para los datos de EEG y eye tracking. Esto implica que los sensores de entrada pueden incluir hardware de monitoreo de actividad cerebral (EEG) y seguimiento de movimientos oculares (eye tracking), pero nuevamente, no se proporciona información explícita sobre otros sensores.\n\nEn resumen, aunque no se menciona explícitamente qué sensores se utilizaron en Multimodal Observer, se puede inferir que los sensores de entrada pueden incluir hardware de teclado, ratón, EEG y eye tracking.",
    "No tengo información suficiente para responder a la pregunta específica. La información disponible es sobre el dispositivo MindWave y su conexión con TGC (ThinkGear Connector) y MO (Multimodal Observer), pero no hay detalles sobre cómo se utiliza NeuroSky MindWave en MultimodalObserver.",
    "No, según el contexto proporcionado, no se menciona específicamente que se utilice Rapid Application Development (RAD) en NEURONE. Sin embargo, se hace referencia a la importancia del prototipado y la creación de una base de código concreta y funcional para la plataforma, lo cual puede estar relacionado con enfoques de desarrollo agil como RAD, que enfatizan la creación rápida de prototipos funcionales",
]

system_answers_hybrid = [
    "Sí, según la información proporcionada, BITalino es una de las herramientas utilizadas como API o SDK para la captura de señales del cuerpo en Multimodal Observer",
    "Según el texto proporcionado, el software Multimodal Observer (MO) se desarrolló utilizando la metodología de desarrollo ágil conocida como DAD (Discipline of Agile Development). Esta metodología se divide en tres etapas: concepción, construcción y transición. El desarrollo del sistema se realizó de forma incremental, a lo largo de varias iteraciones.",
    "Según la información proporcionada en el contexto, los sensores utilizados en Multimodal Observer son: Cámara Web (para capturar datos desde la cámara Web). Dispositivo MindWave NeuroSky (sensores EEG). Dispositivo The Eye Tribe (tracker de ojos). Es importante destacar que no se mencionan otros sensores específicos, como sensores fisiológicos, micrófonos, cámaras de video, sensores de profundidad, etc. Solo se menciona la existencia de estos sensores como posibles fuentes de datos para el proyecto Multimodal Observer.",
    "Sí, se utiliza NeuroSky MindWave en Multimodal Observer (MO). Según el contexto proporcionado, MO es una herramienta diseñada para capturar y visualizar datos multimodales, y se implementan extensiones para la captura de datos desde dispositivos externos como el EEG MindWave y el eye tracker The Eye Tribe. La extensión US-12 Capturar datos desde el sensor Mind Wave menciona explícitamente que se conecta a la API del sensor MindWave para almacenar los datos de las ondas cerebrales captadas. Esto sugiere que MO utiliza NeuroSky MindWave como dispositivo de captura de datos EEG. Además, la tabla 3.10 muestra que el prototipo P07 tiene una descripción que incluye la visualización de la posición del mouse y la captura desde dispositivos EEG MindWave y eye tracker The Eye Tribe. Esto también sugiere que MO se utiliza con NeuroSky MindWave para esta función. En resumen, según el contexto proporcionado, se utiliza NeuroSky MindWave en Multimodal Observer (MO) para capturar datos de EEG y conectarlos a la visualización del mouse y otros dispositivos externos.",
    "Sí, se utiliza Rapid Application Development (RAD) en NEURONE. Según el texto proporcionado, la metodología utilizada para desarrollar NEURONE es un acercamiento combinado inspirado en RAD y Extreme Programming (XP). Además, la gestión de tareas se inspira en la metodología Kanban.",
]


for i in range(0, len(expected_answers)):

    embeddings_vectorial = model.encode(
        [system_answers_vectorial[i], expected_answers[i]]
    )
    embeddings_hybrid = model.encode([system_answers_hybrid[i], expected_answers[i]])

    similarity_vectorial = cosine_similarity(
        [embeddings_vectorial[0]], [embeddings_vectorial[1]]
    )

    similarity_hybrid = cosine_similarity(
        [embeddings_hybrid[0]], [embeddings_hybrid[1]]
    )

    reference = word_tokenize(expected_answers[i])
    candidate_vectorial = word_tokenize(system_answers_vectorial[i])
    candidate_hybrid = word_tokenize(system_answers_hybrid[i])

    score_vectorial = meteor_score([reference], candidate_vectorial)
    score_hybrid = meteor_score([reference], candidate_hybrid)

    print(f"\nFor the question: {expected_answers[i]}:\n")
    print(
        f"Cosine similarity vectorial: {similarity_vectorial[0][0]:.4f}\nCosine similarity hybrid: {similarity_hybrid[0][0]:.4f}"
    )

    print(
        f"Meteor score vectorial: {score_vectorial:.4f}\nMeteor score hybrid: {score_hybrid:.4f}"
    )
    print("______________________")
