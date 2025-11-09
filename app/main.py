import numpy as np

def numpy_operations():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    array_1d = np.array([1,2,3,4])
    array_2d = np.array([[1,2, 3],
                         [3,4, 5]])    
    array_3d = np.array([[[1,2,3],[4,5,6],[7,8,9]],
                         [[5,6,7],[8,9, 10],[11,12,13]],
                         [[9,10,11],[12,13,14],[15,16,17]]])

    print(np.__version__)

    print("Sum of arrays:", a + b)

    print("1D Numpy Array:", array_1d)

    print("2D Numpy Array:", array_2d)
    print("2D Array Dimensions:", array_2d.ndim)
    print("Array Shape:", array_2d.shape)
    print("Array Data Type:", array_2d.dtype)

    print("3D Numpy Array:", array_3d)
    print("3D Array Dimensions:", array_3d.ndim)
    print("3D Array Shape:", array_3d.shape)

    # Chain indexing
    element = array_3d[1][2][0]  # [matrix 1, row 2, column 0]
    print("Accessed Element:", element)

    # Multidimensional indexing - faster than chain indexing
    el = array_3d[1,2,0]
    print("Accessed Element (Multidimensional indexing):", el)

def main():
    numpy_operations()
   

if __name__ == "__main__":
    main()


