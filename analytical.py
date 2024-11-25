import sympy as sp
import numpy as np
'''
expressions calculated in mathematic and taken over
'''

def eigenvalue1(Bx, By, x, y, l):
    # Compute the terms inside the square root
    term1 = 4 * Bx**2 + 4 * By**2 + l**2 + 4 * x**2 + 4 * y**2
    term2 = 8 * np.sqrt((Bx**2 + By**2) * (x**2 + y**2))
    
    # Eigenvalue expression
    eigenvalue = -(1 / 2) * np.sqrt(term1 - term2)
    
    return eigenvalue


def eigenvector1(Bx, By, x, y, l):
    # Compute the common terms
    sqrt_term = np.sqrt((Bx**2 + By**2) * (x**2 + y**2))
    term1 = 4 * Bx**2 + 4 * By**2 + l**2 + 4 * x**2 + 4 * y**2
    term2 = 8 * np.sqrt((Bx**2 + By**2) * (x**2 + y**2))
    
    # Eigenvector expression (real and imaginary parts are handled)
    term1_vector = -(((-2 * Bx * x**2 + 2j * By * x**2 + 1j * Bx * l * y + By * l * y
                      - 2 * Bx * y**2 + 2j * By * y**2 + 2 * Bx * sqrt_term - 2j * By * sqrt_term 
                      + Bx * x * np.sqrt(term1 - term2) - 1j * By * x * np.sqrt(term1 - term2)) /
                      (2 * Bx**2 * y + 2 * By**2 * y + 1j * l * sqrt_term - 2 * y * sqrt_term)))
    
    term2_vector = -((1j * (2 * Bx**2 * x + 2 * By**2 * x - 2 * x * sqrt_term + sqrt_term * np.sqrt(term1 - term2))) /
                     (-2j * Bx**2 * y - 2j * By**2 * y + l * sqrt_term + 2j * y * sqrt_term))
    
    term3_vector = -((8 * 1j * Bx * l * x + 8 * By * l * x - 8 * Bx * y * np.sqrt(term1 - term2)
                      + 8 * 1j * By * y * np.sqrt(term1 - term2)) /
                     (-4 * 1j * Bx**2 * l - 4 * 1j * By**2 * l - 1j * l**3 - 4 * 1j * l * x**2
                      - 8 * Bx**2 * y - 8 * By**2 * y + 2 * l**2 * y + 8 * x**2 * y - 4 * 1j * l * y**2
                      + 8 * y**3 + 1j * l * (4 * Bx**2 + 4 * By**2 + l**2 + 4 * x**2 + 4 * y**2 - 8 * np.sqrt((Bx**2 + By**2) * (x**2 + y**2)))
                      - 2 * y * (4 * Bx**2 + 4 * By**2 + l**2 + 4 * x**2 + 4 * y**2 - 8 * np.sqrt((Bx**2 + By**2) * (x**2 + y**2)))))

    # Return the eigenvector as a list of components
    eigenvector = [term1_vector, term2_vector, term3_vector, 1]
    
    return eigenvector


def eigenvalue2(Bx, By, x, y, l):
    # Compute the terms inside the square root
    term1 = 4 * Bx**2 + 4 * By**2 + l**2 + 4 * x**2 + 4 * y**2
    term2 = 8 * np.sqrt((Bx**2 + By**2) * (x**2 + y**2))
    
    # Eigenvalue expression
    eigenvalue = (1 / 2) * np.sqrt(term1 - term2)
    
    return eigenvalue

def eigenvector2(Bx, By, x, y, l):
    # Compute the common terms
    sqrt_term = np.sqrt((Bx**2 + By**2) * (x**2 + y**2))
    term1 = 4 * Bx**2 + 4 * By**2 + l**2 + 4 * x**2 + 4 * y**2
    term2 = 8 * np.sqrt((Bx**2 + By**2) * (x**2 + y**2))
    
    # Eigenvector expression (real and imaginary parts are handled)
    term1_vector = -(((-2 * Bx * x**2 + 2j * By * x**2 + 1j * Bx * l * y + By * l * y
                      - 2 * Bx * y**2 + 2j * By * y**2 + 2 * Bx * sqrt_term - 2j * By * sqrt_term
                      - Bx * x * np.sqrt(term1 - term2) + 1j * By * x * np.sqrt(term1 - term2)) /
                      (2 * Bx**2 * y + 2 * By**2 * y + 1j * l * sqrt_term - 2 * y * sqrt_term)))
    
    term2_vector = -((1j * (-2 * Bx**2 * x - 2 * By**2 * x + 2 * x * sqrt_term + sqrt_term * np.sqrt(term1 - term2))) /
                     (-2j * Bx**2 * y - 2j * By**2 * y + l * sqrt_term + 2j * y * sqrt_term))
    
    term3_vector = -((8 * 1j * Bx * l * x + 8 * By * l * x + 8 * Bx * y * np.sqrt(term1 - term2)
                      - 8 * 1j * By * y * np.sqrt(term1 - term2)) /
                     (-4 * 1j * Bx**2 * l - 4 * 1j * By**2 * l - 1j * l**3 - 4 * 1j * l * x**2
                      - 8 * Bx**2 * y - 8 * By**2 * y + 2 * l**2 * y + 8 * x**2 * y - 4 * 1j * l * y**2
                      + 8 * y**3 + 1j * l * (4 * Bx**2 + 4 * By**2 + l**2 + 4 * x**2 + 4 * y**2 - 8 * np.sqrt((Bx**2 + By**2) * (x**2 + y**2)))
                      - 2 * y * (4 * Bx**2 + 4 * By**2 + l**2 + 4 * x**2 + 4 * y**2 - 8 * np.sqrt((Bx**2 + By**2) * (x**2 + y**2)))))

    # Return the eigenvector as a list of components
    eigenvector = [term1_vector, term2_vector, term3_vector, 1]
    
    return eigenvector

def eigenvalue3(Bx, By, x, y, l):
    # Compute the terms inside the square root
    term1 = 4 * Bx**2 + 4 * By**2 + l**2 + 4 * x**2 + 4 * y**2
    term2 = 8 * np.sqrt((Bx**2 + By**2) * (x**2 + y**2))
    
    # Eigenvalue expression
    eigenvalue = -(1 / 2) * np.sqrt(term1 + term2)
    
    return eigenvalue

def eigenvector3(Bx, By, x, y, l):
    # Compute the common terms
    sqrt_term = np.sqrt((Bx**2 + By**2) * (x**2 + y**2))
    term1 = 4 * Bx**2 + 4 * By**2 + l**2 + 4 * x**2 + 4 * y**2
    term2 = 8 * np.sqrt((Bx**2 + By**2) * (x**2 + y**2))
    
    # Eigenvector expression (real and imaginary parts are handled)
    term1_vector = -(((-2 * Bx * x**2 + 2j * By * x**2 + 1j * Bx * l * y + By * l * y
                      - 2 * Bx * y**2 + 2j * By * y**2 - 2 * Bx * sqrt_term + 2j * By * sqrt_term
                      + Bx * x * np.sqrt(term1 + term2) - 1j * By * x * np.sqrt(term1 + term2)) /
                      (2 * Bx**2 * y + 2 * By**2 * y - 1j * l * sqrt_term + 2 * y * sqrt_term)))
    
    term2_vector = -((1j * (-2 * Bx**2 * x - 2 * By**2 * x - 2 * x * sqrt_term + sqrt_term * np.sqrt(term1 + term2))) /
                     (2j * Bx**2 * y + 2j * By**2 * y + l * sqrt_term + 2j * y * sqrt_term))
    
    term3_vector = -((8 * 1j * Bx * l * x + 8 * By * l * x - 8 * Bx * y * np.sqrt(term1 + term2)
                      + 8 * 1j * By * y * np.sqrt(term1 + term2)) /
                     (-4 * 1j * Bx**2 * l - 4 * 1j * By**2 * l - 1j * l**3 - 4 * 1j * l * x**2
                      - 8 * Bx**2 * y - 8 * By**2 * y + 2 * l**2 * y + 8 * x**2 * y - 4 * 1j * l * y**2
                      + 8 * y**3 + 1j * l * (4 * Bx**2 + 4 * By**2 + l**2 + 4 * x**2 + 4 * y**2 + 8 * np.sqrt((Bx**2 + By**2) * (x**2 + y**2)))
                      - 2 * y * (4 * Bx**2 + 4 * By**2 + l**2 + 4 * x**2 + 4 * y**2 + 8 * np.sqrt((Bx**2 + By**2) * (x**2 + y**2)))))

    # Return the eigenvector as a list of components
    eigenvector = [term1_vector, term2_vector, term3_vector, 1]
    
    return eigenvector

def eigenvalue4(Bx, By, x, y, l):
    # Compute the terms inside the square root
    term1 = 4 * Bx**2 + 4 * By**2 + l**2 + 4 * x**2 + 4 * y**2
    term2 = 8 * np.sqrt((Bx**2 + By**2) * (x**2 + y**2))
    
    # Eigenvalue expression
    eigenvalue = (1 / 2) * np.sqrt(term1 + term2)
    
    return eigenvalue



def eigenvector4(Bx, By, x, y, l):
    # Compute the common terms
    sqrt_term = np.sqrt((Bx**2 + By**2) * (x**2 + y**2))
    term1 = 4 * Bx**2 + 4 * By**2 + l**2 + 4 * x**2 + 4 * y**2
    term2 = 8 * np.sqrt((Bx**2 + By**2) * (x**2 + y**2))
    
    # Eigenvector expression (real and imaginary parts are handled)
    term1_vector = -(((-2 * Bx * x**2 + 2j * By * x**2 + 1j * Bx * l * y + By * l * y
                      - 2 * Bx * y**2 + 2j * By * y**2 - 2 * Bx * sqrt_term + 2j * By * sqrt_term
                      + Bx * x * np.sqrt(term1 + term2) - 1j * By * x * np.sqrt(term1 + term2)) /
                      (2 * Bx**2 * y + 2 * By**2 * y - 1j * l * sqrt_term + 2 * y * sqrt_term)))
    
    term2_vector = -((1j * (2 * Bx**2 * x + 2 * By**2 * x + 2 * x * sqrt_term + sqrt_term * np.sqrt(term1 + term2))) /
                     (2j * Bx**2 * y + 2j * By**2 * y + l * sqrt_term + 2j * y * sqrt_term))
    
    term3_vector = -((8 * 1j * Bx * l * x + 8 * By * l * x + 8 * Bx * y * np.sqrt(term1 + term2)
                      - 8 * 1j * By * y * np.sqrt(term1 + term2)) /
                     (-4 * 1j * Bx**2 * l - 4 * 1j * By**2 * l - 1j * l**3 - 4 * 1j * l * x**2
                      - 8 * Bx**2 * y - 8 * By**2 * y + 2 * l**2 * y + 8 * x**2 * y - 4 * 1j * l * y**2
                      + 8 * y**3 + 1j * l * (4 * Bx**2 + 4 * By**2 + l**2 + 4 * x**2 + 4 * y**2 + 8 * np.sqrt((Bx**2 + By**2) * (x**2 + y**2)))
                      - 2 * y * (4 * Bx**2 + 4 * By**2 + l**2 + 4 * x**2 + 4 * y**2 + 8 * np.sqrt((Bx**2 + By**2) * (x**2 + y**2)))))

    # Return the eigenvector as a list of components
    eigenvector = [term1_vector, term2_vector, term3_vector, 1]
    
    return eigenvector


