"""
Utilidades para traducción de texto
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)


def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    use_api: bool = False
) -> str:
    """
    Traduce texto entre idiomas
    
    Args:
        text: Texto a traducir
        source_lang: Idioma origen
        target_lang: Idioma destino
        use_api: Si True, usa API externa (Google Translate)
        
    Returns:
        Texto traducido
    """
    if not use_api:
        # Traducción simple (diccionario local)
        return _translate_local(text, source_lang, target_lang)
    else:
        # Usar API externa
        return _translate_with_api(text, source_lang, target_lang)


def _translate_local(
    text: str,
    source_lang: str,
    target_lang: str
) -> str:
    """Traducción local con diccionario"""
    
    # Diccionario LSC <-> ASL
    lsc_to_asl = {
        'Hola': 'Hello',
        'Gracias': 'Thank you',
        'Por favor': 'Please',
        'Sí': 'Yes',
        'No': 'No',
        'Ayuda': 'Help',
        'Familia': 'Family',
        'Amor': 'Love',
        'Casa': 'Home',
        'Trabajo': 'Work'
    }
    
    asl_to_lsc = {v: k for k, v in lsc_to_asl.items()}
    
    if source_lang == 'LSC' and target_lang == 'ASL':
        return lsc_to_asl.get(text, text)
    elif source_lang == 'ASL' and target_lang == 'LSC':
        return asl_to_lsc.get(text, text)
    
    return text


def _translate_with_api(
    text: str,
    source_lang: str,
    target_lang: str
) -> str:
    """Traducción usando API externa"""
    try:
        from deep_translator import GoogleTranslator
        
        # Mapear códigos de idioma
        lang_map = {
            'LSC': 'es',  # Español
            'ASL': 'en'   # Inglés
        }
        
        source = lang_map.get(source_lang, 'es')
        target = lang_map.get(target_lang, 'en')
        
        translator = GoogleTranslator(source=source, target=target)
        translated = translator.translate(text)
        
        return translated
        
    except Exception as e:
        logger.error(f"Error en traducción API: {e}")
        return text


def get_language_name(lang_code: str) -> str:
    """
    Obtiene nombre completo del idioma
    
    Args:
        lang_code: Código del idioma (LSC, ASL)
        
    Returns:
        Nombre completo
    """
    names = {
        'LSC': 'Lengua de Señas Colombiana',
        'ASL': 'American Sign Language'
    }
    
    return names.get(lang_code, lang_code)


def get_language_flag(lang_code: str) -> str:
    """
    Obtiene emoji de bandera del idioma
    
    Args:
        lang_code: Código del idioma
        
    Returns:
        Emoji de bandera
    """
    flags = {
        'LSC': '🇨🇴',
        'ASL': '🌍'
    }
    
    return flags.get(lang_code, '🏳️')