def celsius_to_fahrenheit(celsius: float) -> float:
    """
    Converte a temperatura de Celsius para Fahrenheit.
    
    :param celsius: Temperatura em graus Celsius.
    :return: Temperatura em graus Fahrenheit.
    """
    return (celsius * 9/5) + 32

def fahrenheit_to_celsius(fahrenheit: float) -> float:
    """
    Converte a temperatura de Fahrenheit para Celsius.
    
    :param fahrenheit: Temperatura em graus Fahrenheit.
    :return: Temperatura em graus Celsius.
    """
    return (fahrenheit - 32) * 5/9
