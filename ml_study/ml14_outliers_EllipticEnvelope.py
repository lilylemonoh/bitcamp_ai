import numpy as np
from sklearn.covariance import EllipticEnvelope # 이상치 탐지

outliers_data = np.array([
    -50, -10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 50, 100]
)

print(outliers_data.shape) #(21,)
outliers_data = outliers_data.reshape(-1, 1)
print(outliers_data.shape) #(21, 1)

#EllipticEnvelope 적용
outliers = EllipticEnvelope(contamination=.3)
outliers.fit(outliers_data)
result = outliers.predict(outliers_data)

print(result)
print(result.shape)



# # 상관계수 히트맵(heatmap)
# import matplotlib.pyplot as plt
# plt.boxplot(outlers_loc)

# plt.show()