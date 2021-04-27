from typing import Optional

__parameter_algorithm_separator: str = "___"
__parameter_conditions_separator: str = "|"


def encode_parameter(parameter_name: str, algorithm_name: str,
                     algorithm_separator: str = __parameter_algorithm_separator) -> str:
    """
    Encode parameter name with algorithm name to a string
    If input parameter name is already encoded it is not encoded again, otherwise inputs are encoded

    Args:
        parameter_name (str): algorithm name that parameter belongs to
        algorithm_name (str): name of the parameter to encode
        algorithm_separator (str): separator to use

    Returns:
        Encoded string with parameter name and algorithm name
    """

    if is_parameter_encoded(parameter_name):
        return parameter_name

    return f"{parameter_name}{algorithm_separator}{algorithm_name}"


def decode_parameter(encoded_parameter: str, algorithm_name: str,
                     algorithm_separator: str = __parameter_algorithm_separator) -> Optional[str]:
    """
    Check if encoded payload is correct and decode it
    Args:
        encoded_parameter (str): encoded string
        algorithm_name (str): expected algorithm name
        algorithm_separator (str): expected parameter separator (should be same as in encoding)

    Returns:
        Decoded parameter if it belongs to expected algorithm, None otherwise
    """
    str_ls = encoded_parameter.split(algorithm_separator)
    return str_ls[0] if algorithm_name == str_ls[-1] else None


def is_parameter_encoded(parameter_string: str,
                         algorithm_separator: str = __parameter_algorithm_separator,
                         conditions_separator: str = __parameter_conditions_separator) -> bool:
    """
    Check if parameter string is encoded
    Args:
        parameter_string (str): string with parameter, can be encoded
        algorithm_separator (str): algorithm separator used in encoding
        conditions_separator (str): conditions separator

    Returns:
        True if parameter_string is encoded using StringUtils, False otherwise
    """
    return (parameter_string.find(algorithm_separator) != -1) or (parameter_string.find(conditions_separator) != -1)
