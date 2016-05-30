import data.utilities as d

utilities = d.DataClass(path='\\csv\\test')

X_id, X_raw = utilities.load_x()
Theta_raw = utilities.load_theta()
y_id, y_raw = utilities.load_y()

X = utilities.init_random(len(X_raw), 5)
Theta = utilities.init_random(len(Theta_raw), 5)

y = utilities.init_y(y_raw, X_id, Theta_raw, 1000, 1000)
R = utilities.get_R(y)

print(R.sum())
