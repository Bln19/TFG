import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
import os
from urllib.parse import urlparse, parse_qs

# Tabla de datos
data = {
    # "Nº": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 
    #     29, 30, 31, 32, 33, 34, 35],
    "Nº": [1, 2, 3, 45],
    # "Youtuber y Streamer": ["Luisito Comunica", "El Rubius", "DrossRotzank", "Yuya", "Fernanfloo", "WindyGirk",
    #                         "Mariale", "Werevertumorro", "HolaSoyGerman", "AuronPlay", "EnchufeTV", "Luisa Fernanda W",
    #                         "Mox", "Ciencia de sofá", "La Divaza", "Julioprofe", "Alejo Igoa", "Xuri Fenton", "Caeli",
    #                         "Gabriel Herrera", "Claudia Nicolasa", "Álex Otaola", "Gibby", "Roberto Martínez",
    #                         "Jorge Cremades", "Sergio Mohadeb", "El Robot de Platón", "DotCSV", "La Pulla", "Rolando Morán",
    #                         "Cocina con Carmen", "Isa Marcial", "CdeCiencia", "Roció Vidal", "Juan Ramón Rayo"],

    "Youtuber y Streamer": ["Luisito Comunica", "El Rubius", "DrossRotzank", "Yuya"],

    # "País": ["México", "España", "Venezuela", "México", "El Salvador", "Pánama", "Venezuela", "México", "Chile",
    #         "España", "Ecuador", "Colombia", "Perú", "Bolivia", "Venezuela", "Colombia", "Argentina", "Honduras",
    #         "México", "Paraguay", "Perú", "Cuba", "México", "Bolivia", "España", "Argentina", "Perú", "España",
    #         "Colombia", "República Dominicana", "España", "España", "España", "España", "España"],
    "País": ["México", "España", "Venezuela", "México"],
    
    # "Región": ["-", "Madrid", "-", "-", "-", "-", "-", "-", "-", "Cataluña", "-", "-", "-", "-", "-", "-", "-", "-", "-",
    #         "-", "-", "-", "-", "-", "Comunidad Valenciana", "-", "-", "Canarias", "-",  "-", "Andalucía", "Extremadura", "Murcia",
    #         "-", "Comunidad Valenciana"],
    "Región": ["-", "Madrid", "-", "-"],
    
    # "Temática principal": ["Viajes y cultura", "Gaming y Entretenimiento", "Terror y misterio", "Belleza y lifestyle", 
    #                     "Gaming", "Gaming y vlogs", "Belleza y compras", "Humor y entretenimiento", "Comedia",
    #                     "Humor y crítica social", "Sketches de comedia", "Entretenimiento", "Gaming y comedia",
    #                     "Conceptos científicos", "Comedia y Vlogs", "Educación", "Vlogs y Retos", "Música y lifestyle",
    #                     "Lifestyle", "Sociedad y política", "Psicología y crítica social", "Política y sociología cubana",
    #                     "Entretenimiento infantil", "Economía y finanzas", "Comedia", "Derecho y temas legales", "Divulgación científica y social",
    #                     "Inteligencia artificial", "Política y sociedad", "Cultura y sociedad", "Cocina y gastronomía", "Tecnología y reviews",
    #                     "Ciencia y Educación", "Divulgación científica y crítica", "Economía"],
    "Temática principal": ["Viajes y cultura", "Gaming y Entretenimiento", "Terror y misterio", "Belleza y lifestyle"],
    
    "Enlaces vídeos": [
        ["https://youtu.be/2at1uc0LfQY?si=0RcbBnS2Zqyq_kVS", "https://youtu.be/w-pIBtP21vc?si=9ovVo40_kTxTVBmg"],
        ["https://youtu.be/Q0xVOwNmx9w?si=WGn-kk2cKCXGDjTe", "https://youtu.be/bXSIpq11MfE?si=3X789dy18JK9h3pv"],
        ["https://youtu.be/6aswzDXxj7U?si=y2_uCSsXV3ti_1wj", "https://youtu.be/q7Lo6SLfapg?si=FDU9DBwnFTrjI6-E"],
        ["https://youtu.be/wd_MQ3os0M0?si=gxzakow7YAFuoYqq", "https://youtu.be/ldATTcJpIpE?si=eRyoDjuWnJa7NTeF"],
        # ["https://youtu.be/fMPB9ZjZF0g?si=obdO7HVCZqMCi_gO", "https://youtu.be/CKBt580csvU?si=cREAs4Chql5pEc7v"],
        # ["https://youtu.be/NdTQGFIORV0?si=KWFEsKPHvynW9apA", "https://youtu.be/wPsk6M6CFlI?si=WfztCyEeYmYbypvb"],
        # ["https://youtu.be/OBJr7-_iI0A?si=IN5c3KmKbfe11htK", "https://youtu.be/v-_MBQjFIk8?si=A51PMlZj_kHH0sBq"],
        # ["https://youtu.be/EcaAVHkjnOw?si=agL4B3KOfpKzEZEp", "https://youtu.be/A8akvovIc0Q?si=vStbwEUz6XVIh7_e"],
        # ["https://youtu.be/oz0hdMkQ_9w?si=gdAS5N2rqOBx4iPt", "https://youtu.be/mheiH6YmC7s?si=pYeUk1SE6dnyRcz5"],
        # ["https://youtu.be/Uo-ZphDU06Y?si=jD91jOWZCyfVoH8D", "https://youtu.be/5Y8lyOVmU20?si=KrEMbvxg5qFRbHjp"],
        # ["https://youtu.be/2vlASNiEDNk?si=_JkX7sINUkQ6Juie", "https://youtu.be/rhg3vh16o7I?si=9fN3NgFyw3Xp0v4t"],
        # ["https://youtu.be/kosTST0CE88?si=pOpzjnUsG85kv9SA", "https://youtu.be/FHnFfL-loXo?si=Pim7CpWqOLQJRqvZ"],
        # ["https://youtu.be/RoI_6G0-e9E?si=afUakuUJX-crRA5R", "https://youtu.be/RrJCJPmKNqg?si=dwLorJWsYN1nZRqa"],
        # ["https://youtu.be/HrueTlm8gX8?si=rrZbRllHoScbXxhJ", "https://youtu.be/DAnGjQRMOvk?si=A-YTIMwLkG17YyJ0"],
        # ["https://youtu.be/43ZWeFvtRcE?si=9V3sYoPvP5OwVws6", "https://youtu.be/noRVR1SBinE?si=whae_aXNL2WRTuSr"],
        # ["https://youtu.be/kKwNiAZZXVM?si=cKjdPH8p-ifUWCVe", "https://youtu.be/W-O1MPtm0qo?si=SyAL03msDGKJN67q"],
        # ["https://youtu.be/aEtLVkJYmBo?si=QoPbj0pmRn4hwD3Z", "https://youtu.be/A-ZVfe5p080?si=2P02EYzUhHsVeOgM"],
        # ["https://youtu.be/8yr18iPPELA?si=fnHQe6cZ7MJDN4Pa", "https://youtu.be/0QCLRTUKClI?si=7hrkLubvsnoXt25d"],
        # ["https://youtu.be/N2T0r28mAIA?si=Mkx382bya-HwJLdk", "https://youtu.be/YbWLdSObG54?si=LwdSLAoDORxkbENk"],
        # ["https://youtu.be/FX2aUAMxa3E?si=BpdgapxhqWo0kpFj", "https://youtu.be/Lb6bQ6mjpQc?si=r3DR-oZA_C7zal0L"],
        # ["https://youtu.be/LURhwxEHohU?si=avLQOALecxqiWNHf", "https://youtu.be/2VpKVtXxuYU?si=1VjPmVZdC23iaR3o"],
        # ["https://youtu.be/BJt4owS7YXM?si=-oWQFnfhRLv__AhP", "https://youtu.be/VuCsqOr7DpA?si=RFo6yFjEr6-ovx2m"],
        # ["https://youtu.be/rTQI7ttBEGI?si=m_tPAljSRUauUGgg", "https://youtu.be/TL5RzAhpuow?si=wlzu4vV89j0A0mMo"],
        # ["https://youtu.be/e9_rQdxKAbA?si=77Lk0Gp4LTPIdS_Z", "https://youtu.be/fheaILJJABQ?si=XRpVqfWpl8-AkSxF"],
        # ["https://youtu.be/rX35wLHnpvk?si=OBO2WGDym8IspbBL", "https://youtu.be/-6xEt5wYm8M?si=UNN4ircWj4xDx9CO"],
        # ["https://youtu.be/qbxsOeYNxts?si=KdNfLrGY1aAvq1qR", "https://youtu.be/lWN4u-OVnNE?si=p4LPsdaDS9iVKx1O"],
        # ["https://youtu.be/rGAygFbY2XY?si=WOyJWFfaEhxXw_SX", "https://youtu.be/HUE60JFt7hU?si=pMIXH6VFYLUokSqo"],
        # ["https://youtu.be/PtR3MI89bK4?si=9eqWQnC8l2IDO5bX", "https://youtu.be/rKJdwLkPPPM?si=spo_cLgAb7H8Sals"],
        # ["https://youtu.be/33RF8J_q6vk?si=oqCaSgq732KGakNg", "https://youtu.be/yW5FxPvXSjQ?si=GgF1AodKAZcz1e8x"],
        # ["https://youtu.be/tdmBG3dk8TI?si=HIHF5pvnHb8DlpD_", "https://youtu.be/_oIqVuk9vVo?si=H2U6EaspMAguO2-v"],
        # ["https://youtu.be/QENL_KUwGVg?si=auBzQ09ML7y6HErL", "https://youtu.be/luouqcGOjGw?si=nb0ap9PW2BPhMOZ_"],
        # ["https://youtu.be/jQClQM_t3cY?si=6Dv0PVTGRQrSLNkN", "https://youtu.be/7oXNd43zHQw?si=mGgS6hdZ9KWF_OzL"],
        # ["https://youtu.be/YHfXev6ExfU?si=FQNMks1w2YRfOr3o", "https://youtu.be/CpVpJpYVe3I?si=-HtLWoRaBg0u7sFB"],
        # ["https://youtu.be/Et--2TuBM8E?si=KBoKyAAkNkxYgHVm", "https://youtu.be/qEd4NjU2EeQ?si=ZsphYf4Bbels1KBK"],
        # ["https://youtu.be/Bivx6NJs4cI?si=IVz9NoQEVvgfnV5-", "https://youtu.be/x0SuEp7DdpI?si=LC3uxsrGbn6siBuf"]
    ]
}

# Crear DataFrame
df = pd.DataFrame(data)

# Directorio para guardar las transcripciones
output_dir = '/home/jovyan/work/data/transcripciones'
# os.makedirs(output_dir, exist_ok=True)

# Función para extraer la transcripción de un video
def get_transcription(video_id):
    try:
        # Obtener la transcripción especificando el idioma preferido
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['es'])
        # Unir todos los textos en una sola cadena
        transcript_text = " ".join([entry['text'] for entry in transcript])
        return transcript_text
    except Exception as e:
        return str(e)

# Función para extraer el video_id de un enlace de YouTube
def extract_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        return parse_qs(parsed_url.query).get('v', [None])[0]
    return None

# Extraer transcripciones y guardarlas en archivos
for index, row in df.iterrows():
    for link in row["Enlaces vídeos"]:
        video_id = extract_video_id(link)
        if video_id:
            transcription = get_transcription(video_id)
            # Guardar la transcripción en un archivo
            with open(os.path.join(output_dir, f"{row['Youtuber y Streamer']}_{video_id}.txt"), "w", encoding="utf-8") as f:
                f.write(transcription)

print("Transcripciones extraidas correctamente y guardadas en la carpeta 'transcripciones' (en la carpeta de notebooks).")
