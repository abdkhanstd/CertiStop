"""Download the two public datasets into data_raw/.

Malaga treadmill tests: PhysioNet treadmill-exercise-cardioresp 1.0.1 (doi:10.13026/7ezk-j442),
PhysioNet Contributor Review Health Data License 1.5.0.
Dublin graded incremental tests: Zenodo record 10841412 (doi:10.5281/zenodo.10841412), CC BY 4.0.
Usage: python certistop/download_data.py
"""
import hashlib, os, urllib.request, zipfile

FILES = [
    ('https://physionet.org/files/treadmill-exercise-cardioresp/1.0.1/subject-info.csv', 'data_raw/malaga/subject-info.csv', None),
    ('https://physionet.org/files/treadmill-exercise-cardioresp/1.0.1/test_measure.csv', 'data_raw/malaga/test_measure.csv', None),
    ('https://zenodo.org/records/10841412/files/data_v2.zip?download=1', 'data_raw/dublin/data_v2.zip', 'f64cb1bf4d66129daed6c1954ea6ca9f'),
]

for url, dst, md5 in FILES:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        print('downloading', url)
        urllib.request.urlretrieve(url, dst)
    if md5:
        h = hashlib.md5(open(dst, 'rb').read()).hexdigest()
        assert h == md5, f'md5 mismatch for {dst}: {h}'
with zipfile.ZipFile('data_raw/dublin/data_v2.zip') as z:
    z.extractall('data_raw/dublin/ex')
print('done')
