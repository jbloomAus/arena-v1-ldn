import random 
import string
from torch.utils.data import Dataset

def is_palindrome(s):
    return s == s[::-1]

def generate_palindrome(s):
        return s + s[::-1]

def generate_random_palindrome(k=12, alphabet=string.ascii_lowercase):
    s = ''.join(random.choices(alphabet, k=k))
    return generate_palindrome(s)

def perturb_palindrome(s):
    s = list(s)
    i = random.randint(0, len(s)-1)
    j = random.randint(0, len(s)-1)
    s[i] = s[j]
    return ''.join(s)

def perturb_palindrom_n_times(s, n):
    for _ in range(n):
        s = perturb_palindrome(s)
    return s

def get_palindrome_distance(s):
    distance = 0
    for i in range(len(s)//2):
        if s[i] != s[-i-1]:
            distance += 1
    return distance

from torch.utils.data import Dataset, DataLoader

class PalindromeDataset(Dataset):
    '''
    dataset = PalindromeDataset(1000, perturb_n_times=8)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    '''
    def __init__(self, n_samples, k=12, perturb_n_times=1, alphabet=string.ascii_lowercase):
        self.n_samples = n_samples
        self.k = k
        self.perturb_n_times = perturb_n_times
        self.ratio_positive = 0.5
        self.alphabet = alphabet
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        if random.random() < self.ratio_positive:
            s = generate_random_palindrome(self.k, alphabet=self.alphabet)
            y = is_palindrome(s)
        else:
            s = generate_random_palindrome(self.k, alphabet=self.alphabet)
            s = perturb_palindrom_n_times(s, self.perturb_n_times)
            y = is_palindrome(s)
        return s, y
