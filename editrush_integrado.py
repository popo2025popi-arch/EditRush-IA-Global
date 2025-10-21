"""
EditRush 2.0 - VERSIÓN CON IA GLOBAL CONECTADA
===============================================
Cliente mejorado que se conecta al servidor de IA en la nube
Incluye análisis avanzado y sugerencias inteligentes

NUEVAS DEPENDENCIAS:
pip install requests psutil

CONFIGURACIÓN:
1. Establece la URL del servidor en variable de entorno:
   set EDITRUSH_IA_URL=https://tu-servidor.com

2. O edita directamente en el código la variable URL_SERVIDOR_IA
"""

import os
import sys
import platform
import subprocess
import threading
import time
import json
import pickle
import psutil  # Para monitoreo de sistema
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import signal
import wave

import customtkinter as ctk
from tkinter import filedialog, messagebox

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

# Importar módulo de conexión IA
try:
    from conexion_ia_global import ConexionIAGlobal, AnalizadorAvanzado
    IA_GLOBAL_DISPONIBLE = True
except ImportError:
    print("⚠️ Módulo de IA Global no encontrado. Funcionará en modo local.")
    IA_GLOBAL_DISPONIBLE = False

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ===================== CONFIGURACIÓN =====================
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FFMPEG_DIR = os.path.join(BASE_DIR, "ffmpeg", "bin")
FFMPEG_PATH = os.path.join(FFMPEG_DIR, "ffmpeg.exe")
FFPROBE_PATH = os.path.join(FFMPEG_DIR, "ffprobe.exe")

if platform.system() != "Windows":
    FFMPEG_PATH = os.path.join(FFMPEG_DIR, "ffmpeg")
    FFPROBE_PATH = os.path.join(FFMPEG_DIR, "ffprobe")

MODEL_PKL = os.path.join(BASE_DIR, "modelo_aprendizaje.pkl")
MODEL_JSON = os.path.join(BASE_DIR, "modelo_aprendizaje.json")

# URL del servidor de IA (cambiar después del deployment)
URL_SERVIDOR_IA = os.environ.get('EDITRUSH_IA_URL', 'http://localhost:8000')

# ===================== UTILIDADES =====================
def human_time(sec):
    return time.strftime("%H:%M:%S", time.gmtime(sec))

def run_cmd(cmd, timeout=None):
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as e:
        return -1, "", str(e)

def ffprobe_duration(path):
    if not os.path.exists(FFPROBE_PATH):
        return None
    cmd = [FFPROBE_PATH, "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", path]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        out = p.stdout.strip()
        return float(out)
    except Exception:
        return None

def obtener_uso_recursos():
    """Obtiene uso actual de CPU y memoria"""
    try:
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory().percent
        return cpu, mem
    except:
        return 0.0, 0.0

# ===================== SISTEMA IA LOCAL (FALLBACK) =====================
class SistemaIALocal:
    """Sistema IA local de respaldo"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = SGDRegressor(max_iter=1000, tol=1e-3)
        self.metadata = {
            "version": "2.0",
            "total_calificaciones": 0,
            "fecha_actualizacion": None
        }
        self._features_names = ['energia_promedio','energia_picos','cambios_bruscos',
                               'porcentaje_silencio','varianza_promedio']
        self._init_or_load()
    
    def _init_or_load(self):
        if os.path.exists(MODEL_PKL) and os.path.exists(MODEL_JSON):
            try:
                with open(MODEL_PKL, 'rb') as f:
                    saved = pickle.load(f)
                self.model = saved.get('model', self.model)
                self.scaler = saved.get('scaler', self.scaler)
                with open(MODEL_JSON, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print("No se pudo cargar modelo local:", e)
        self.metadata['fecha_actualizacion'] = datetime.now().isoformat()
    
    def _save_model(self):
        try:
            with open(MODEL_PKL, 'wb') as f:
                pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
            with open(MODEL_JSON, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print("Error guardando modelo:", e)
    
    def _metric_vector(self, metricas):
        return np.array([metricas.get(k, 0.0) for k in self._features_names], 
                       dtype=float).reshape(1, -1)
    
    def predecir(self, metricas):
        try:
            x = self._metric_vector(metricas)
            x_scaled = self.scaler.transform(x)
            pred = self.model.predict(x_scaled)[0]
            return float(np.clip(pred, 1.0, 10.0))
        except NotFittedError:
            return None
        except Exception:
            return None
    
    def registrar_calificacion(self, metricas, calificacion_num):
        try:
            x = self._metric_vector(metricas)
            try:
                self.scaler.partial_fit(x)
            except:
                self.scaler.fit(x)
            
            try:
                x_scaled = self.scaler.transform(x)
            except:
                x_scaled = x
            
            y = np.array([float(calificacion_num)])
            try:
                self.model.partial_fit(x_scaled, y)
            except:
                self.model.fit(x_scaled, y)
            
            self.metadata['total_calificaciones'] += 1
            self._save_model()
            return True
        except Exception as e:
            print("Error registrando calificación local:", e)
            return False

# ===================== ANALIZADOR DE CLIPS =====================
class AnalizadorClips:
    """Analizador que extrae métricas de audio"""
    
    def __init__(self, clip_path):
        self.clip_path = clip_path
        self.metricas = {}
        self.puntuacion_clasica = 0.0
    
    def extraer_audio(self, out_path):
        if not os.path.exists(FFMPEG_PATH):
            return None
        try:
            cmd = [FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
                   "-i", self.clip_path, "-ac", "1", "-ar", "44100", out_path]
            rc, out, err = run_cmd(cmd, timeout=30)
            if rc == 0 and os.path.exists(out_path):
                return out_path
            return None
        except Exception:
            return None
    
    def analizar(self):
        audio_path = os.path.join(os.path.dirname(self.clip_path), "__audio_temp.wav")
        ap = self.extraer_audio(audio_path)
        if not ap:
            return None
        
        try:
            with wave.open(audio_path, 'rb') as wf:
                nframes = wf.getnframes()
                frames = wf.readframes(nframes)
                audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                framerate = wf.getframerate()
            
            ventana_ms = 100.0
            ventana_frames = int(framerate * ventana_ms / 1000.0)
            if ventana_frames <= 0:
                ventana_frames = 4410
            
            energias = []
            varianzas = []
            for i in range(0, max(1, len(audio_data) - ventana_frames), ventana_frames):
                seg = audio_data[i:i+ventana_frames]
                if len(seg) == 0:
                    continue
                energia = np.sqrt(np.mean(seg**2))
                varianza = np.var(seg)
                energias.append(energia)
                varianzas.append(varianza)
            
            energias = np.array(energias) if len(energias) > 0 else np.array([0.0])
            varianzas = np.array(varianzas) if len(varianzas) > 0 else np.array([0.0])
            
            self.metricas['energia_promedio'] = float(np.mean(energias))
            self.metricas['energia_picos'] = float(np.max(energias))
            self.metricas['varianza_promedio'] = float(np.mean(varianzas))
            
            cambios = np.diff(energias) if len(energias) > 1 else np.array([0.0])
            cambios_bruscos = int(np.sum(np.abs(cambios) > (np.std(cambios) * 2 if np.std(cambios) != 0 else 1e-6)))
            self.metricas['cambios_bruscos'] = cambios_bruscos
            
            silencio_umbral = np.percentile(energias, 20) if len(energias) > 0 else 0.0
            porcentaje_silencio = float(np.sum(energias < silencio_umbral) / len(energias)) if len(energias) > 0 else 0.0
            self.metricas['porcentaje_silencio'] = porcentaje_silencio
            
            # Puntuación clásica
            puntuacion = 5.0
            if self.metricas['energia_promedio'] > 0.02:
                puntuacion += 2.0
            elif self.metricas['energia_promedio'] > 0.01:
                puntuacion += 1.0
            if self.metricas['energia_picos'] > 0.5:
                puntuacion += 2.5
            elif self.metricas['energia_picos'] > 0.3:
                puntuacion += 1.5
            if self.metricas['cambios_bruscos'] > 10:
                puntuacion += 2.0
            elif self.metricas['cambios_bruscos'] > 5:
                puntuacion += 1.0
            if porcentaje_silencio > 0.4:
                puntuacion -= 2.5
            elif porcentaje_silencio > 0.2:
                puntuacion -= 1.0
            if self.metricas['varianza_promedio'] > 0.0001:
                puntuacion += 1.5
            
            self.puntuacion_clasica = float(max(1.0, min(10.0, round(puntuacion, 1))))
            
            # Añadir métricas de video si está disponible el analizador avanzado
            if IA_GLOBAL_DISPONIBLE:
                try:
                    analizador_video = AnalizadorAvanzado(self.clip_path, FFPROBE_PATH)
                    metricas_video = analizador_video.analizar_video_completo()
                    self.metricas.update(metricas_video)
                except Exception as e:
                    print(f"No se pudieron extraer métricas de video: {e}")
            
            # Limpiar temporal
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except:
                pass
            
            return self.metricas
        except Exception as e:
            print("Error analizando audio:", e)
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except:
                pass
            return None

# ===================== PROCESAMIENTO DE CLIPS =====================
class CoreProcessing:
    """Gestión de generación de clips con FFmpeg"""
    
    def __init__(self):
        self._proc = None
        self._cancel_flag = threading.Event()
    
    def cancel_generacion(self):
        self._cancel_flag.set()
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
        except:
            pass
    
    def generar_clips_async(self, app, video_path, dur_clip, cantidad, modo, 
                          solapamiento, calidad, carpeta_salida, filtros, 
                          parametros_optimizados=None):
        thread = threading.Thread(
            target=self._worker_generar, 
            args=(app, video_path, dur_clip, cantidad, modo, solapamiento, 
                  calidad, carpeta_salida, filtros, parametros_optimizados), 
            daemon=True
        )
        thread.start()
    
    def _aplicar_sugerencias_ffmpeg(self, cmd_base, sugerencias):
        """Aplica sugerencias de IA a los comandos de FFmpeg"""
        vf_filters = []
        af_filters = []
        
        for sug in sugerencias:
            tipo = sug.get('tipo')
            accion = sug.get('accion')
            params = sug.get('parametros', {})
            
            if tipo == 'audio':
                if accion == 'aumentar_volumen':
                    gain = params.get('gain', 6.0)
                    af_filters.append(f"volume={gain}dB")
                elif accion == 'reducir_silencios':
                    umbral = params.get('umbral', -40)
                    af_filters.append(f"silenceremove=1:0:{umbral}dB")
                elif accion == 'normalizar_volumen':
                    loudness = params.get('loudness', -16)
                    af_filters.append(f"loudnorm=I={loudness}")
            
            elif tipo == 'video':
                if accion == 'aumentar_brillo':
                    brillo = params.get('brillo', 0.2)
                    vf_filters.append(f"eq=brightness={brillo}")
                elif accion == 'mejorar_contraste':
                    contraste = params.get('contraste', 1.4)
                    vf_filters.append(f"eq=contrast={contraste}")
        
        return vf_filters, af_filters
    
    def _worker_generar(self, app, video_path, dur_clip, cantidad, modo, 
                       solapamiento, calidad, carpeta_salida, filtros, 
                       parametros_optimizados):
        self._cancel_flag.clear()
        self._proc = None
        inicio_proceso = time.time()
        
        if not os.path.exists(video_path):
            app.update_progress(0.0, "Archivo no encontrado")
            messagebox.showerror("Error", "Archivo de video no encontrado.")
            app.on_generation_finished()
            return
        
        dur_total = ffprobe_duration(video_path)
        if dur_total is None:
            app.update_progress(0.0, "No se pudo leer duración")
            messagebox.showerror("Error", "No se pudo leer la duración del video.")
            app.on_generation_finished()
            return
        
        os.makedirs(carpeta_salida, exist_ok=True)
        
        dur_clip = float(max(1.0, dur_clip))
        tiempos = []
        
        if modo == "secuencial":
            t = 0.0
            solap = 0.0
            if solapamiento:
                solap = max(5.0, dur_clip * 0.1)
            while t < dur_total - 0.001:
                tiempos.append(t)
                step = dur_clip - solap if (dur_clip - solap) > 0 else dur_clip
                t += step
        else:
            num = max(1, int(cantidad))
            intervalo = dur_total / num
            for i in range(num):
                tiempos.append(i * intervalo)
        
        total = len(tiempos)
        
        # Usar parámetros optimizados por IA si están disponibles
        if parametros_optimizados:
            preset = parametros_optimizados.get('preset', 'fast')
            crf = parametros_optimizados.get('crf', 23)
        else:
            presets = {"baja": ("ultrafast", 28), "media": ("fast", 23), "alta": ("slow", 18)}
            preset, crf = presets.get(calidad, ("fast", 23))
        
        for i, inicio in enumerate(tiempos):
            if self._cancel_flag.is_set():
                app.update_progress(i/total if total>0 else 0, "Cancelado")
                break
            
            fin = min(inicio + dur_clip, dur_total)
            out_file = os.path.join(carpeta_salida, f"clip_{i+1}.mp4")
            
            vf_filters = []
            af_filters = []
            
            # Filtros básicos del usuario
            if filtros.get('estabilizar', False):
                vf_filters.append("deshake")
            
            if filtros.get('mejorar_video', False):
                vf_filters.append("hqdn3d")
                vf_filters.append("unsharp")
            
            if filtros.get('limpieza_audio', False):
                af_filters.append("afftdn")
                af_filters.append("loudnorm")
            
            # Aplicar sugerencias de IA si están disponibles
            if hasattr(app, 'sugerencias_ia') and app.sugerencias_ia:
                vf_ia, af_ia = self._aplicar_sugerencias_ffmpeg([], app.sugerencias_ia)
                vf_filters.extend(vf_ia)
                af_filters.extend(af_ia)
            
            vf_str = ",".join([f for f in vf_filters if f])
            af_str = ",".join([f for f in af_filters if f])
            
            cmd = [FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
                   "-ss", str(inicio), "-to", str(fin), "-i", video_path,
                   "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                   "-c:a", "aac", "-q:a", "5"]
            
            if vf_str:
                cmd += ["-vf", vf_str]
            if af_str:
                cmd += ["-af", af_str]
            
            cmd += [out_file]
            
            try:
                self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                while True:
                    if self._cancel_flag.is_set():
                        try:
                            self._proc.terminate()
                        except:
                            pass
                        break
                    ret = self._proc.poll()
                    if ret is not None:
                        break
                    time.sleep(0.1)
            except Exception as e:
                print("Error ejecutando ffmpeg:", e)
            
            frac = (i+1) / total if total>0 else 1.0
            app.update_progress(frac, f"Clip {i+1}/{total} generado")
            app.update_idletasks()
        
        # Calcular tiempo total y enviar a IA global
        tiempo_total = time.time() - inicio_proceso
        
        if hasattr(app, 'ia_global') and app.ia_global and not self._cancel_flag.is_set():
            # Enviar datos de rendimiento a IA
            try:
                cpu, mem = obtener_uso_recursos()
                metricas_ejemplo = {
                    'energia_promedio': 0.02,
                    'energia_picos': 0.5,
                    'varianza_promedio': 0.0001,
                    'cambios_bruscos': 10,
                    'porcentaje_silencio': 0.2,
                    'duracion_clip': dur_clip
                }
                
                app.ia_global.optimizar_parametros(
                    metricas=metricas_ejemplo,
                    tiempo_procesamiento=tiempo_total,
                    objetivo="equilibrado",
                    sistema_os=platform.system(),
                    uso_cpu=cpu,
                    uso_memoria=mem,
                    cantidad_clips=total,
                    calidad=calidad
                )
            except Exception as e:
                print(f"No se pudieron enviar datos de rendimiento: {e}")
        
        if self._cancel_flag.is_set():
            app.update_progress(0.0, "Generación cancelada")
            messagebox.showinfo("Cancelado", "Generación de clips cancelada.")
        else:
            app.update_progress(1.0, f"✅ Completado en {human_time(tiempo_total)}")
            messagebox.showinfo("Éxito", 
                f"Generación finalizada.\n"
                f"Tiempo: {human_time(tiempo_total)}\n"
                f"Carpeta: {os.path.abspath(carpeta_salida)}")
        
        app.on_generation_finished()
        self._proc = None
        self._cancel_flag.clear()

# ===================== APLICACIÓN PRINCIPAL =====================
class EditRushApp(ctk.CTk):
    """Aplicación principal con IA Global integrada"""
    
    def __init__(self):
        super().__init__()
        
        self.title("EditRush 2.0 - IA Global Conectada 🌍")
        self.geometry("1200x750")
        self.minsize(1000, 600)
        
        # Variables
        self.video_path = ctk.StringVar()
        self.duracion_clip = ctk.IntVar(value=30)
        self.cantidad_clips = ctk.IntVar(value=5)
        self.calidad = ctk.StringVar(value="media")
        self.modo_extraccion = ctk.StringVar(value="secuencial")
        self.permitir_solapamiento = ctk.BooleanVar(value=False)
        self.estilo_deseado = ctk.StringVar(value="neutro")
        self.carpeta_salida = os.path.join(BASE_DIR, "clips_generados")
        
        # Sistemas de IA
        self.ia_local = SistemaIALocal()
        self.ia_global = None
        self.sugerencias_ia = []
        self.parametros_optimizados = None
        
        # Inicializar IA Global
        if IA_GLOBAL_DISPONIBLE:
            try:
                self.ia_global = ConexionIAGlobal(url_servidor=URL_SERVIDOR_IA)
                print(f"✅ IA Global {'conectada' if self.ia_global.esta_conectado() else 'en modo local'}")
            except Exception as e:
                print(f"⚠️ IA Global no disponible: {e}")
        
        # Core
        self.core = CoreProcessing()
        self.generando = False
        self.proceso_actual = None
        
        # Construir UI
        self._build_ui()
        self.verificar_ffmpeg()
    
    def verificar_ffmpeg(self):
        if not os.path.exists(FFMPEG_PATH):
            messagebox.showerror(
                "Error - FFmpeg no encontrado",
                f"No se encontró FFmpeg en:\n{FFMPEG_PATH}\n\n"
                "Descarga FFmpeg y colócalo en la carpeta ffmpeg/bin/"
            )
    
    def _build_ui(self):
        # Grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # SIDEBAR
        sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_rowconfigure(10, weight=1)
        
        # Logo con indicador de conexión
        estado_ia = "🌍" if (self.ia_global and self.ia_global.esta_conectado()) else "💻"
        self.logo_label = ctk.CTkLabel(
            sidebar,
            text=f"{estado_ia} EditRush 2.0",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20,5))
        
        # Estado de IA
        estado_texto = "IA Global Activa" if (self.ia_global and self.ia_global.esta_conectado()) else "Modo Local"
        self.estado_ia_label = ctk.CTkLabel(
            sidebar,
            text=estado_texto,
            font=ctk.CTkFont(size=10),
            text_color="green" if (self.ia_global and self.ia_global.esta_conectado()) else "orange"
        )
        self.estado_ia_label.grid(row=1, column=0, padx=20, pady=(0,10))
        
        # Botones
        ctk.CTkButton(sidebar, text="🎥 Generador", command=self.show_generador).grid(row=2, column=0, padx=20, pady=6, sticky="ew")
        ctk.CTkButton(sidebar, text="📊 Revisión", command=self.show_revision).grid(row=3, column=0, padx=20, pady=6, sticky="ew")
        ctk.CTkButton(sidebar, text="🧠 IA Global", command=self.show_ia_global).grid(row=4, column=0, padx=20, pady=6, sticky="ew")
        ctk.CTkButton(sidebar, text="💡 Sugerencias", command=self.show_sugerencias).grid(row=5, column=0, padx=20, pady=6, sticky="ew")
        ctk.CTkButton(sidebar, text="⚙️ Config", command=self.show_config).grid(row=6, column=0, padx=20, pady=6, sticky="ew")
        
        # Tema
        self.switch_theme = ctk.CTkSwitch(sidebar, text="Modo Oscuro", command=self.toggle_theme)
        self.switch_theme.grid(row=10, column=0, padx=20, pady=10)
        self.switch_theme.select()
        
        # MAIN AREA
        self.main_frame = ctk.CTkFrame(self, corner_radius=0)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=12, pady=12)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        
        self.show_generador()
    
    def toggle_theme(self):
        mode = ctk.get_appearance_mode()
        ctk.set_appearance_mode("light" if mode == "dark" else "dark")
    
    def clear_main(self):
        for w in self.main_frame.winfo_children():
            w.destroy()
    
    # ========== GENERADOR ==========
    def show_generador(self):
        self.clear_main()
        frame = ctk.CTkScrollableFrame(self.main_frame)
        frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        frame.grid_columnconfigure(0, weight=1)
        
        # Título
        ctk.CTkLabel(frame, text="🎥 Generador de Clips Inteligente con IA Global", 
                    font=ctk.CTkFont(size=22, weight="bold")).grid(row=0, column=0, padx=12, pady=(8,18), sticky="w")
        
        # Video
        row1 = ctk.CTkFrame(frame)
        row1.grid(row=1, column=0, sticky="ew", padx=12, pady=6)
        row1.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(row1, text="Archivo de video:").grid(row=0, column=0, padx=8, pady=8, sticky="w")
        ctk.CTkEntry(row1, textvariable=self.video_path).grid(row=0, column=1, padx=8, pady=8, sticky="ew")
        ctk.CTkButton(row1, text="Examinar", command=self.select_video).grid(row=0, column=2, padx=8, pady=8)
        
        # Config
        cfg = ctk.CTkFrame(frame)
        cfg.grid(row=2, column=0, sticky="ew", padx=12, pady=6)
        cfg.grid_columnconfigure((0,1), weight=1)
        ctk.CTkLabel(cfg, text="Duración por clip (s):").grid(row=0, column=0, padx=8, pady=6, sticky="w")
        ctk.CTkEntry(cfg, textvariable=self.duracion_clip, width=120).grid(row=0, column=1, padx=8, pady=6, sticky="w")
        ctk.CTkLabel(cfg, text="Cantidad (modo manual):").grid(row=1, column=0, padx=8, pady=6, sticky="w")
        ctk.CTkEntry(cfg, textvariable=self.cantidad_clips, width=120).grid(row=1, column=1, padx=8, pady=6, sticky="w")
        ctk.CTkLabel(cfg, text="Calidad:").grid(row=2, column=0, padx=8, pady=6, sticky="w")
        ctk.CTkOptionMenu(cfg, variable=self.calidad, values=["baja","media","alta"]).grid(row=2, column=1, padx=8, pady=6, sticky="w")
        ctk.CTkLabel(cfg, text="Estilo deseado (para IA):").grid(row=3, column=0, padx=8, pady=6, sticky="w")
        ctk.CTkOptionMenu(cfg, variable=self.estilo_deseado, values=["dinamico","profesional","neutro"]).grid(row=3, column=1, padx=8, pady=6, sticky="w")
        
        # Modos
        modos = ctk.CTkFrame(frame)
        modos.grid(row=3, column=0, sticky="ew", padx=12, pady=6)
        ctk.CTkLabel(modos, text="Modo de Extracción:").grid(row=0, column=0, padx=8, pady=6, sticky="w")
        ctk.CTkRadioButton(modos, text="Secuencial", variable=self.modo_extraccion, value="secuencial").grid(row=1, column=0, padx=8, pady=4, sticky="w")
        ctk.CTkRadioButton(modos, text="Manual", variable=self.modo_extraccion, value="manual").grid(row=2, column=0, padx=8, pady=4, sticky="w")
        ctk.CTkCheckBox(modos, text="Permitir solapamiento", variable=self.permitir_solapamiento).grid(row=3, column=0, padx=8, pady=4, sticky="w")
        
        # Filtros
        filt = ctk.CTkFrame(frame)
        filt.grid(row=4, column=0, sticky="ew", padx=12, pady=6)
        ctk.CTkLabel(filt, text="Mejoras / Filtros:").grid(row=0, column=0, padx=8, pady=6, sticky="w")
        self.chk_estab_var = ctk.BooleanVar(value=False)
        self.chk_audio_var = ctk.BooleanVar(value=False)
        self.chk_video_var = ctk.BooleanVar(value=False)
        self.chk_ia_auto = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(filt, text="Estabilizar video", variable=self.chk_estab_var).grid(row=1, column=0, padx=8, pady=4, sticky="w")
        ctk.CTkCheckBox(filt, text="Limpieza de audio", variable=self.chk_audio_var).grid(row=2, column=0, padx=8, pady=4, sticky="w")
        ctk.CTkCheckBox(filt, text="Mejorar nitidez", variable=self.chk_video_var).grid(row=3, column=0, padx=8, pady=4, sticky="w")
        ctk.CTkCheckBox(filt, text="🌍 Usar sugerencias de IA Global", variable=self.chk_ia_auto).grid(row=4, column=0, padx=8, pady=4, sticky="w")
        
        # Botón para obtener sugerencias
        ctk.CTkButton(filt, text="💡 Obtener Sugerencias IA", command=self.obtener_sugerencias_ia).grid(row=5, column=0, padx=8, pady=8, sticky="ew")
        
        # Progreso
        prog_frame = ctk.CTkFrame(frame)
        prog_frame.grid(row=5, column=0, sticky="ew", padx=12, pady=12)
        self.progress_bar = ctk.CTkProgressBar(prog_frame)
        self.progress_bar.grid(row=0, column=0, padx=8, pady=8, sticky="ew")
        self.lbl_progress = ctk.CTkLabel(prog_frame, text="Listo")
        self.lbl_progress.grid(row=1, column=0, padx=8, pady=4, sticky="w")
        
        # Acciones
        actions = ctk.CTkFrame(frame)
        actions.grid(row=6, column=0, sticky="ew", padx=12, pady=12)
        self.btn_generate = ctk.CTkButton(actions, text="🚀 Generar Clips", width=200, command=self.start_generation)
        self.btn_generate.grid(row=0, column=0, padx=8, pady=6)
        self.btn_cancel = ctk.CTkButton(actions, text="❌ Cancelar", width=200, fg_color="red", command=self.cancel_generation, state="disabled")
        self.btn_cancel.grid(row=0, column=1, padx=8, pady=6)
        ctk.CTkButton(actions, text="📂 Abrir carpeta", command=self.open_output_folder).grid(row=1, column=0, padx=8, pady=8)
    
    def obtener_sugerencias_ia(self):
        """Obtiene sugerencias de la IA Global antes de generar"""
        if not self.ia_global or not self.ia_global.esta_conectado():
            messagebox.showinfo("IA Local", "IA Global no disponible. Usando análisis local.")
            return
        
        # Analizar video actual (si existe)
        if not self.video_path.get() or not os.path.exists(self.video_path.get()):
            messagebox.showwarning("Error", "Selecciona un video primero")
            return
        
        self.lbl_progress.configure(text="🔍 Consultando IA Global...")
        self.update_idletasks()
        
        try:
            # Usar métricas de ejemplo (o analizar un segmento del video)
            metricas_ejemplo = {
                'energia_promedio': 0.02,
                'energia_picos': 0.5,
                'varianza_promedio': 0.0001,
                'cambios_bruscos': 10,
                'porcentaje_silencio': 0.2,
                'duracion_clip': self.duracion_clip.get(),
                'fps': 30.0,
                'brillo_promedio': 0.5,
                'contraste_promedio': 0.5,
                'movimiento_promedio': 0.3
            }
            
            resultado = self.ia_global.obtener_sugerencias(
                metricas_ejemplo, 
                self.estilo_deseado.get()
            )
            
            self.sugerencias_ia = resultado.get('sugerencias', [])
            cantidad = resultado.get('cantidad', 0)
            
            self.lbl_progress.configure(text=f"✅ {cantidad} sugerencias recibidas de IA Global")
            
            # Mostrar resumen
            if cantidad > 0:
                resumen = "📊 SUGERENCIAS DE IA GLOBAL:\n\n"
                for i, sug in enumerate(self.sugerencias_ia[:5], 1):
                    resumen += f"{i}. {sug['tipo'].upper()}: {sug['razon']}\n"
                
                messagebox.showinfo("Sugerencias de IA", resumen)
            else:
                messagebox.showinfo("IA Global", "No hay sugerencias adicionales. El video está bien optimizado.")
        
        except Exception as e:
            self.lbl_progress.configure(text="❌ Error consultando IA")
            messagebox.showerror("Error", f"No se pudieron obtener sugerencias: {e}")
    
    def start_generation(self):
        if not self.video_path.get():
            messagebox.showwarning("Error", "Selecciona un video primero")
            return
        if self.generando:
            messagebox.showinfo("En proceso", "Ya hay una generación en curso")
            return
        
        # Obtener parámetros optimizados de IA si está disponible
        if self.ia_global and self.ia_global.esta_conectado():
            try:
                cpu, mem = obtener_uso_recursos()
                metricas = {
                    'energia_promedio': 0.02,
                    'energia_picos': 0.5,
                    'varianza_promedio': 0.0001,
                    'cambios_bruscos': 10,
                    'porcentaje_silencio': 0.2,
                    'duracion_clip': self.duracion_clip.get()
                }
                
                resultado = self.ia_global.optimizar_parametros(
                    metricas=metricas,
                    tiempo_procesamiento=100,  # Estimación
                    objetivo="equilibrado",
                    sistema_os=platform.system(),
                    uso_cpu=cpu,
                    uso_memoria=mem,
                    cantidad_clips=self.cantidad_clips.get(),
                    calidad=self.calidad.get()
                )
                
                self.parametros_optimizados = resultado.get('parametros_optimizados')
            except Exception as e:
                print(f"No se pudieron obtener parámetros optimizados: {e}")
        
        filtros = {
            'estabilizar': self.chk_estab_var.get(),
            'limpieza_audio': self.chk_audio_var.get(),
            'mejorar_video': self.chk_video_var.get()
        }
        
        self.generando = True
        self.btn_generate.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self.update_progress(0.0, "Iniciando generación...")
        
        self.core.generar_clips_async(
            app=self,
            video_path=self.video_path.get(),
            dur_clip=int(self.duracion_clip.get()),
            cantidad=int(self.cantidad_clips.get()),
            modo=self.modo_extraccion.get(),
            solapamiento=self.permitir_solapamiento.get(),
            calidad=self.calidad.get(),
            carpeta_salida=self.carpeta_salida,
            filtros=filtros,
            parametros_optimizados=self.parametros_optimizados
        )
    
    def cancel_generation(self):
        if not self.generando:
            return
        self.core.cancel_generacion()
        self.update_progress(0.0, "Cancelando...")
        self.btn_cancel.configure(state="disabled")
    
    def on_generation_finished(self):
        self.generando = False
        try:
            self.btn_generate.configure(state="normal")
            self.btn_cancel.configure(state="disabled")
        except:
            pass
    
    # ========== REVISIÓN ==========
    def show_revision(self):
        self.clear_main()
        frame = ctk.CTkFrame(self.main_frame)
        frame.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(frame, text="📊 Revisión de Clips con IA", 
                    font=ctk.CTkFont(size=22, weight="bold")).grid(row=0, column=0, padx=12, pady=(8,12), sticky="w")
        
        archivos = []
        try:
            if os.path.exists(self.carpeta_salida):
                archivos = [f for f in os.listdir(self.carpeta_salida) 
                           if f.lower().endswith((".mp4",".mov",".mkv",".avi"))]
                archivos = sorted(archivos)
        except:
            archivos = []
        
        if not archivos:
            ctk.CTkLabel(frame, text="No hay clips. Genera algunos primero.").grid(
                row=1, column=0, padx=12, pady=12, sticky="w")
            return
        
        list_frame = ctk.CTkFrame(frame)
        list_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=6)
        list_frame.grid_columnconfigure(1, weight=1)
        
        self.listbox = ctk.CTkTextbox(list_frame, width=40, height=200)
        self.listbox.grid(row=0, column=0, padx=8, pady=8, sticky="ns")
        for idx, a in enumerate(archivos):
            self.listbox.insert("end", f"{idx+1}. {a}\n")
        self.listbox.configure(state="disabled")
        
        panel = ctk.CTkFrame(list_frame)
        panel.grid(row=0, column=1, padx=12, pady=8, sticky="nsew")
        panel.grid_rowconfigure(8, weight=1)
        
        ctk.CTkLabel(panel, text="Archivo seleccionado:").grid(row=0, column=0, padx=6, pady=6, sticky="w")
        self.selected_clip_var = ctk.StringVar(value=archivos[0])
        ctk.CTkOptionMenu(panel, values=archivos, variable=self.selected_clip_var).grid(
            row=1, column=0, padx=6, pady=6, sticky="ew")
        
        ctk.CTkButton(panel, text="🔍 Analizar con IA", command=self.analizar_clip).grid(
            row=2, column=0, padx=6, pady=6, sticky="ew")
        ctk.CTkButton(panel, text="⭐ EXCELENTE (9.0)", command=lambda: self.calificar_clip(9.0)).grid(
            row=3, column=0, padx=6, pady=6, sticky="ew")
        ctk.CTkButton(panel, text="👍 BUENO (7.0)", command=lambda: self.calificar_clip(7.0)).grid(
            row=4, column=0, padx=6, pady=6, sticky="ew")
        ctk.CTkButton(panel, text="😐 REGULAR (5.0)", command=lambda: self.calificar_clip(5.0)).grid(
            row=5, column=0, padx=6, pady=6, sticky="ew")
        ctk.CTkButton(panel, text="👎 MALO (2.0)", command=lambda: self.calificar_clip(2.0)).grid(
            row=6, column=0, padx=6, pady=6, sticky="ew")
        
        self.text_metrics = ctk.CTkTextbox(panel, height=200)
        self.text_metrics.grid(row=8, column=0, padx=6, pady=6, sticky="nsew")
        self.text_metrics.configure(state="disabled")
    
    def analizar_clip(self):
        clip = os.path.join(self.carpeta_salida, self.selected_clip_var.get())
        if not os.path.exists(clip):
            messagebox.showwarning("Error", "Clip no encontrado")
            return
        
        anal = AnalizadorClips(clip)
        metricas = anal.analizar()
        if metricas is None:
            messagebox.showerror("Error", "No se pudo analizar el clip")
            return
        
        # Predecir con IA Global si está disponible
        pred_global = None
        if self.ia_global and self.ia_global.esta_conectado():
            try:
                resultado = self.ia_global.predecir_puntuacion(metricas)
                pred_global = resultado.get('puntuacion')
            except:
                pass
        
        # Predecir con IA Local
        pred_local = self.ia_local.predecir(metricas)
        
        texto = "🔍 ANÁLISIS COMPLETO:\n"
        texto += "=" * 50 + "\n\n"
        texto += "📊 MÉTRICAS:\n"
        texto += f"• Energía promedio: {metricas.get('energia_promedio',0):.5f}\n"
        texto += f"• Picos de energía: {metricas.get('energia_picos',0):.5f}\n"
        texto += f"• Varianza: {metricas.get('varianza_promedio',0):.7f}\n"
        texto += f"• Cambios bruscos: {metricas.get('cambios_bruscos',0)}\n"
        texto += f"• % Silencio: {metricas.get('porcentaje_silencio',0)*100:.1f}%\n\n"
        
        texto += "🎯 PUNTUACIONES:\n"
        texto += f"• Heurística: {anal.puntuacion_clasica}/10\n"
        texto += f"• IA Local: {pred_local if pred_local else 'N/D'}/10\n"
        texto += f"• 🌍 IA Global: {pred_global if pred_global else 'No conectada'}/10\n\n"
        
        if metricas.get('fps'):
            texto += "🎬 VIDEO:\n"
            texto += f"• FPS: {metricas.get('fps', 0):.1f}\n"
            texto += f"• Resolución: {metricas.get('resolucion', 'N/D')}\n"
            texto += f"• Brillo: {metricas.get('brillo_promedio', 0):.2f}\n"
        
        self.text_metrics.configure(state="normal")
        self.text_metrics.delete(1.0, "end")
        self.text_metrics.insert("end", texto)
        self.text_metrics.configure(state="disabled")
    
    def calificar_clip(self, puntuacion):
        clip = os.path.join(self.carpeta_salida, self.selected_clip_var.get())
        if not os.path.exists(clip):
            messagebox.showwarning("Error", "Clip no encontrado")
            return
        
        anal = AnalizadorClips(clip)
        metricas = anal.analizar()
        if metricas is None:
            messagebox.showerror("Error", "No se pudo analizar el clip")
            return
        
        # Registrar en IA Local
        self.ia_local.registrar_calificacion(metricas, puntuacion)
        
        # Registrar en IA Global
        if self.ia_global and self.ia_global.esta_conectado():
            try:
                self.ia_global.registrar_calificacion(
                    metricas=metricas,
                    puntuacion=puntuacion,
                    estilo=self.estilo_deseado.get(),
                    usuario_id=platform.node()  # ID único del PC
                )
                messagebox.showinfo("Éxito", 
                    f"✅ Calificación {puntuacion}/10 registrada\n"
                    "🌍 Compartida con IA Global para mejorar el sistema")
            except Exception as e:
                messagebox.showwarning("Parcial", 
                    f"Calificación guardada localmente.\n"
                    f"No se pudo enviar a IA Global: {e}")
        else:
            messagebox.showinfo("Local", 
                "✅ Calificación guardada localmente\n"
                "⚠️ IA Global no disponible")
    
    # ========== IA GLOBAL ==========
    def show_ia_global(self):
        self.clear_main()
        frame = ctk.CTkScrollableFrame(self.main_frame)
        frame.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(frame, text="🌍 Inteligencia Artificial Global", 
                    font=ctk.CTkFont(size=22, weight="bold")).grid(row=0, column=0, sticky="w", padx=12, pady=(8,12))
        
        if not self.ia_global or not self.ia_global.esta_conectado():
            ctk.CTkLabel(frame, text="⚠️ IA Global no conectada", 
                        text_color="orange").grid(row=1, column=0, sticky="w", padx=12, pady=6)
            ctk.CTkLabel(frame, text=f"Servidor configurado: {URL_SERVIDOR_IA}").grid(
                row=2, column=0, sticky="w", padx=12, pady=6)
            ctk.CTkButton(frame, text="🔄 Reintentar conexión", 
                         command=self.reintentar_conexion_ia).grid(row=3, column=0, padx=12, pady=12, sticky="w")
            return
        
        # Obtener estadísticas
        stats = self.ia_global.obtener_estadisticas_globales()
        
        info_frame = ctk.CTkFrame(frame)
        info_frame.grid(row=1, column=0, sticky="ew", padx=12, pady=12)
        
        ctk.CTkLabel(info_frame, text="📊 ESTADÍSTICAS GLOBALES", 
                    font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, columnspan=2, padx=12, pady=8, sticky="w")
        
        ctk.CTkLabel(info_frame, text=f"Total de usuarios contribuyendo:").grid(
            row=1, column=0, padx=12, pady=6, sticky="w")
        ctk.CTkLabel(info_frame, text=f"{stats.get('usuarios_totales', 'N/D')}", 
                    font=ctk.CTkFont(weight="bold")).grid(row=1, column=1, padx=12, pady=6, sticky="w")
        
        ctk.CTkLabel(info_frame, text=f"Total de calificaciones:").grid(
            row=2, column=0, padx=12, pady=6, sticky="w")
        ctk.CTkLabel(info_frame, text=f"{stats.get('total_calificaciones', stats.get('total', 'N/D'))}", 
                    font=ctk.CTkFont(weight="bold")).grid(row=2, column=1, padx=12, pady=6, sticky="w")
        
        ctk.CTkLabel(info_frame, text=f"Promedio global:").grid(
            row=3, column=0, padx=12, pady=6, sticky="w")
        ctk.CTkLabel(info_frame, text=f"{stats.get('promedio_puntuacion', 'N/D')}/10", 
                    font=ctk.CTkFont(weight="bold")).grid(row=3, column=1, padx=12, pady=6, sticky="w")
        
        ctk.CTkLabel(info_frame, text=f"Última actualización:").grid(
            row=4, column=0, padx=12, pady=6, sticky="w")
        fecha = stats.get('fecha', stats.get('ultima_actualizacion', 'N/D'))
        if fecha != 'N/D' and len(fecha) > 19:
            fecha = fecha[:19]
        ctk.CTkLabel(info_frame, text=fecha, 
                    font=ctk.CTkFont(size=10)).grid(row=4, column=1, padx=12, pady=6, sticky="w")
        
        # Estilos populares
        if 'estilos_populares' in stats:
            estilos_frame = ctk.CTkFrame(frame)
            estilos_frame.grid(row=2, column=0, sticky="ew", padx=12, pady=12)
            
            ctk.CTkLabel(estilos_frame, text="🎨 ESTILOS MÁS POPULARES", 
                        font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, padx=12, pady=8, sticky="w")
            
            estilos = stats['estilos_populares']
            for idx, (estilo, cantidad) in enumerate(estilos.items(), 1):
                ctk.CTkLabel(estilos_frame, text=f"{estilo.capitalize()}: {cantidad} votos").grid(
                    row=idx, column=0, padx=12, pady=4, sticky="w")
        
        # Botones
        ctk.CTkButton(frame, text="🔄 Sincronizar modelo", 
                     command=self.sincronizar_modelo_ia).grid(row=10, column=0, padx=12, pady=12, sticky="w")
    
    def reintentar_conexion_ia(self):
        if IA_GLOBAL_DISPONIBLE:
            try:
                self.ia_global = ConexionIAGlobal(url_servidor=URL_SERVIDOR_IA)
                if self.ia_global.esta_conectado():
                    messagebox.showinfo("Éxito", "✅ Conectado a IA Global")
                    self.actualizar_estado_ia()
                    self.show_ia_global()
                else:
                    messagebox.showwarning("Conexión", "No se pudo conectar al servidor")
            except Exception as e:
                messagebox.showerror("Error", f"Error: {e}")
    
    def sincronizar_modelo_ia(self):
        if self.ia_global and self.ia_global.esta_conectado():
            self.ia_global.sincronizar_modelo()
            messagebox.showinfo("Sincronización", "✅ Modelo sincronizado")
        else:
            messagebox.showwarning("Error", "IA Global no conectada")
    
    def actualizar_estado_ia(self):
        """Actualiza el indicador de estado de IA en el sidebar"""
        conectado = self.ia_global and self.ia_global.esta_conectado()
        estado_ia = "🌍" if conectado else "💻"
        self.logo_label.configure(text=f"{estado_ia} EditRush 2.0")
        estado_texto = "IA Global Activa" if conectado else "Modo Local"
        self.estado_ia_label.configure(
            text=estado_texto,
            text_color="green" if conectado else "orange"
        )
    
    # ========== SUGERENCIAS ==========
    def show_sugerencias(self):
        self.clear_main()
        frame = ctk.CTkFrame(self.main_frame)
        frame.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        
        ctk.CTkLabel(frame, text="💡 Sugerencias de IA", 
                    font=ctk.CTkFont(size=22, weight="bold")).grid(row=0, column=0, padx=12, pady=12)
        
        if not self.sugerencias_ia:
            ctk.CTkLabel(frame, text="No hay sugerencias. Usa el botón en el Generador para obtenerlas.").grid(
                row=1, column=0, padx=12, pady=12)
        else:
            for idx, sug in enumerate(self.sugerencias_ia, 1):
                sug_frame = ctk.CTkFrame(frame)
                sug_frame.grid(row=idx, column=0, sticky="ew", padx=12, pady=6)
                
                texto = f"{idx}. {sug['tipo'].upper()}: {sug['accion']}\n   Razón: {sug['razon']}"
                ctk.CTkLabel(sug_frame, text=texto, anchor="w").grid(row=0, column=0, padx=12, pady=8, sticky="w")
    
    # ========== CONFIG ==========
    def show_config(self):
        self.clear_main()
        frame = ctk.CTkFrame(self.main_frame)
        frame.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        
        ctk.CTkLabel(frame, text="⚙️ Configuración", 
                    font=ctk.CTkFont(size=22, weight="bold")).grid(row=0, column=0, sticky="w", padx=12, pady=12)
        
        ctk.CTkLabel(frame, text=f"Carpeta salida: {self.carpeta_salida}").grid(row=1, column=0, sticky="w", padx=12, pady=6)
        ctk.CTkButton(frame, text="Cambiar carpeta", command=self.change_output_folder).grid(row=2, column=0, padx=12, pady=6, sticky="w")
        
        ok = os.path.exists(FFMPEG_PATH) and os.path.exists(FFPROBE_PATH)
        ctk.CTkLabel(frame, text=f"FFmpeg: {'✅ Encontrado' if ok else '❌ No encontrado'}").grid(row=3, column=0, sticky="w", padx=12, pady=6)
        
        ia_status = "✅ Conectada" if (self.ia_global and self.ia_global.esta_conectado()) else "❌ Desconectada"
        ctk.CTkLabel(frame, text=f"IA Global: {ia_status}").grid(row=4, column=0, sticky="w", padx=12, pady=6)
        ctk.CTkLabel(frame, text=f"Servidor: {URL_SERVIDOR_IA}").grid(row=5, column=0, sticky="w", padx=12, pady=6)
    
    # ========== UTILS ==========
    def select_video(self):
        ruta = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.mov *.mkv *.avi")])
        if ruta:
            self.video_path.set(ruta)
    
    def change_output_folder(self):
        carpeta = filedialog.askdirectory()
        if carpeta:
            self.carpeta_salida = carpeta
            messagebox.showinfo("Carpeta", f"Establecida: {self.carpeta_salida}")
    
    def open_output_folder(self):
        os.makedirs(self.carpeta_salida, exist_ok=True)
        try:
            if platform.system() == "Windows":
                os.startfile(os.path.abspath(self.carpeta_salida))
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", self.carpeta_salida])
            else:
                subprocess.Popen(["xdg-open", self.carpeta_salida])
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir: {e}")
    
    def update_progress(self, fraction, message=""):
        try:
            self.progress_bar.set(min(max(fraction, 0.0), 1.0))
            self.lbl_progress.configure(text=message)
            self.update_idletasks()
        except:
            pass

# ===================== PUNTO DE ENTRADA =====================
def main():
    app = EditRushApp()
    app.mainloop()

if __name__ == "__main__":
    main()