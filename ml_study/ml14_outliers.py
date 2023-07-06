import numpy as np

outliers_data = np.array([
    -50, -10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 50, 100]
)

def outliers(data_out): 
    q1, q2, q3 = np.percentile(data_out, [25, 50, 75])
    print('1사분위 :', q1)
    print('2사분위 :', q2)
    print('3사분위 :', q3)
    
    iqr = q3 - q1
    print('IQR : ', iqr)
    
    lower_bound = q1 - (iqr*1.5)
    upper_bound = q3 + (iqr*1.5)
    print('lower_bound :', lower_bound)
    print('upper_bound :', upper_bound)
    
    return np.where((data_out > upper_bound) | (data_out < lower_bound))

outlers_loc = outliers(outliers_data)
print('이상치의 위치 : ', outlers_loc)

# 상관계수 히트맵(heatmap)
import matplotlib.pyplot as plt
plt.boxplot(outlers_loc)

plt.show()