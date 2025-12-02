import sounddevice as sd
from scipy.io.wavfile import write
import speech_recognition as sr
import tempfile, os, random
from datetime import datetime
import webbrowser

# --- opcional: instalar googletrans ---
# pip install googletrans==4.0.0-rc1
from googletrans import Translator

SRATE = 16000     # tasa de muestreo
DUR = 5           # segundos

print("Grabando... habla ahora!")
audio = sd.rec(int(DUR*SRATE), samplerate=SRATE, channels=1, dtype='int16')
sd.wait()
print("Listo, procesando...")

# guarda a WAV temporal
tmp_wav = tempfile.mktemp(suffix=".wav")
write(tmp_wav, SRATE, audio)

# reconoce con SpeechRecognition
r = sr.Recognizer()
with sr.AudioFile(tmp_wav) as source:
    data = r.record(source)

try:
    texto = r.recognize_google(data, language="es-ES")
    print("Dijiste:", texto)
    cmd = texto.lower()

    if "hola" in cmd:
        print("¡Hola, bienvenido al curso!")

    elif "abrir google" in cmd:
        webbrowser.open("https://www.google.com")

    elif "hora" in cmd:
        print("Hora actual:", datetime.now().strftime("%H:%M"))

    # --- NUEVOS COMANDOS ---
    elif "clima" in cmd:
        webbrowser.open("https://wttr.in/bogota?format=3")
        print("Mostrando clima en Bogotá...")

    elif "traducir" in cmd:
        frase = cmd.replace("traducir", "").strip()
        if frase:
            traductor = Translator()
            traduccion = traductor.translate(frase, src="es", dest="en")
            print(f"Traducción al inglés: {traduccion.text}")
        else:
            print("No dijiste qué traducir.")

    elif "chiste" in cmd:
        chistes = [
            "¿Por qué los programadores confunden Halloween con Navidad? Porque OCT 31 == DEC 25.",
            "¿Qué le dice un bit al otro? Nos vemos en el bus.",
            "¿Por qué la computadora fue al médico? Porque tenía un virus."
        ]
        print(random.choice(chistes))

    else:
        print("Comando no reconocido.")

except sr.UnknownValueError:
    print("No se entendió el audio.")
except sr.RequestError as e:
    print("Error:", e)
finally:
    if os.path.exists(tmp_wav):
        os.remove(tmp_wav)
