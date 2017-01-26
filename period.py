import numpy as np

def weekends_array(T):
	T1 = 95
	T_we = 48
	T_wd = 120
	weekends = []
	for i in range(T):
		if i > T1 and (i-T1-1)%168<T_we:
			weekends.append(True)
		else:
			weekends.append(False)
	weekends = np.array(weekends)
	workdays = np.invert(weekends)
	return weekends, workdays


def days_array(T):
	T_day = 11
	T_night = 24 - T_day
	T1_night = 7
	days = []
	for i in range(T):
		if i >= T1_night and (i-T1_night)%24 < T_day:
			days.append(True)
		else:
			days.append(False)
	days = np.array(days)
	return days

def nights_array(T):
	TT_day = 14
	TT_night = 24 - TT_day
	TT1_night = 6
	nights = []
	for i in range(T):
		if i >= TT1_night and (i-TT1_night)%24 < TT_day:
			nights.append(False)
		else:
			nights.append(True)
	nights = np.array(nights)
	complet_days = np.invert(nights)
	return nights, complet_days

def regimes(T, t0=1, di=(8,19), ni=(21,7), tw0=95, wl=48):
	"""
	tw0: start position
	wl: weekend length
	"""
	regimes = np.zeros(T, dtype=int)
	for i in range(T):
		time = (t0 + i)%24
		if (i >= tw0 and (i-tw0)%168 < wl):
			# weekend
			regimes[i] = 4
		elif ((time >= ni[0] and time < 24) or (time + 24 >= ni[0] and time < ni[1])):
			# night
			regimes[i] = 1
		elif time >= di[0] and time < di[1]:
			# day
			regimes[i] = 0
		elif (time >= di[1] and time < ni[0]):
			# day-to-night
			regimes[i] = 2
		else:
			# night-to-day
			regimes[i] = 3
	return regimes

# a = np.ones(200)*3
# a[days_array(200)] = 0
# a[nights_array(200)[0]] = 1
# a[weekends_array(200)[0]] = 1
# print(a)
# b = regimes(200)
# print(b)
# print(regimes(200, t0=1, di=(10,20), ni=(23,5)))
# print(regimes(200, t0=1, di=(7,18), ni=(21,5)))
# print(a==b)

# print (regimes(100, di=(8,19), ni=(21,6), tw0=0, wl=48))
