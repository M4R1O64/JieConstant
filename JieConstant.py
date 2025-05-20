import gmpy2
from gmpy2 import mpfr, get_context
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ------------------ 설정 ------------------
MAX_N = 1000
get_context().precision = 300 * 4  # 비트 정밀도, 1자리 ≈ 3.32비트

# ------------------ MPFR 복소수 클래스 ------------------
class MPFRComplex:
    def __init__(self, real, imag):
        self.real = mpfr(real)
        self.imag = mpfr(imag)

    def __add__(self, other):
        return MPFRComplex(self.real + other.real, self.imag + other.imag)

    def __sub__(self, other):
        return MPFRComplex(self.real - other.real, self.imag - other.imag)

    def __mul__(self, other):
        r = self.real * other.real - self.imag * other.imag
        i = self.real * other.imag + self.imag * other.real
        return MPFRComplex(r, i)

    def __truediv__(self, other):
        denom = other.real**2 + other.imag**2
        r = (self.real * other.real + self.imag * other.imag) / denom
        i = (self.imag * other.real - self.real * other.imag) / denom
        return MPFRComplex(r, i)

    def conjugate(self):
        return MPFRComplex(self.real, -self.imag)

    def abs(self):
        return MPFRComplex(gmpy2.sqrt(self.real**2 + self.imag**2), mpfr(0))

    def __repr__(self):
        return f"({self.real} + {self.imag}j)"

# ------------------ 계수 계산 ------------------
factorials = [mpfr(1)]
for i in range(1, MAX_N + 1):
    factorials.append(factorials[-1] * i)

coeff = [MPFRComplex(mpfr(1), mpfr(0))]
for n in range(1, MAX_N + 1):
    fct = mpfr(1)
    for j in range(2 * n - 3, 0, -2):
        fct *= j
    c = ((-1) ** (n + 1)) * fct / (2 ** n * factorials[n])
    coeff.append(MPFRComplex(c, mpfr(0)))

# ------------------ 조합 및 DP ------------------
Cb = [[mpfr(0)] * (MAX_N + 1) for _ in range(MAX_N + 1)]
for i in range(MAX_N + 1):
    Cb[i][0] = mpfr(1)
    Cb[i][i] = mpfr(1)
    for j in range(1, i):
        Cb[i][j] = Cb[i - 1][j - 1] + Cb[i - 1][j]

EV = [[mpfr(0)] * (MAX_N + 1) for _ in range(MAX_N + 1)]
DP = [[mpfr(0)] * (MAX_N + 1) for _ in range(MAX_N + 1)]
EV[0][0] = mpfr(1)

print("[+] Calculating EVs...")
for p in tqdm(range(MAX_N + 1)):
    for q in range(MAX_N + 1):
        if (p - q) % 6 == 0 and p + q > 0:
            for i in range(p + 1):
                if i < p:
                    EV[p][q] += Cb[p][i] * DP[i][q]
                else:
                    for j in range(q + 1):
                        if i == p and j == q:
                            continue
                        if (i - j) % 6 == 0:
                            EV[p][q] += Cb[p][i] * Cb[q][j] * EV[i][j]
            EV[p][q] *= mpfr(2) / (3 * (2 ** (p + q) - 1))
        for i in range(q + 1):
            DP[p][q] += Cb[q][i] * EV[p][i]
print("[+] Done!!")

# ------------------ 계산 함수 ------------------
def calculate_value(U: MPFRComplex, a: MPFRComplex):
    A = a / U
    B = a.conjugate() / U.conjugate()

    arrA = [MPFRComplex(1, 0)]
    arrB = [MPFRComplex(1, 0)]
    for i in range(1, MAX_N + 1):
        arrA.append(arrA[-1] * A)
        arrB.append(arrB[-1] * B)

    coeffA = [MPFRComplex(1, 0)]
    coeffB = [MPFRComplex(1, 0)]
    for n in range(1, MAX_N + 1):
        coeffA.append(coeff[n] * arrA[n])
        coeffB.append(coeff[n] * arrB[n])

    total = MPFRComplex(0, 0)
    print("[+] Adding coeffs to calculate value", U, a)
    for i in range(MAX_N + 1):
        for j in range(MAX_N + 1):
            total += coeffA[i] * coeffB[j] * MPFRComplex(EV[i][j], 0)
    return total * U.abs()

# ------------------ 병렬 처리 대상 함수 ------------------
def calc_worker(args):
    U, p = args
    X = MPFRComplex(0.5, 0)
    result = calculate_value(X + U * X * X, MPFRComplex(0.25, 0))
    return result * MPFRComplex(p, 0)

# ------------------ 실행부 ------------------
if __name__ == "__main__":
    pi6 = MPFRComplex(0.5, gmpy2.sqrt(3)/2)
    P6 = [MPFRComplex(1, 0)]
    for i in range(1, 6):
        P6.append(P6[-1] * pi6)

    p1 = mpfr(1) / 3
    p2 = mpfr(1) / 9
    arr = [(MPFRComplex(0, 0), p1)]
    for i in range(6):
        if i != 3:
            arr.append((P6[i], p2))

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(calc_worker, arr)

    total = MPFRComplex(0, 0)
    for r in results:
        total = total + r

    ans = total.real * 18 / 17 * mpfr("0.8")
    print(ans, " is ans")
