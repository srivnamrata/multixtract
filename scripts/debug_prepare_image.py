import zipfile
from pathlib import Path
from multixtract.filters import ImageFilterPipeline

p = Path('C:/Users/Z0019386/AppData/Local/Temp/test_docx_debug/test.docx')
with zipfile.ZipFile(str(p), 'r') as zf:
    names = zf.namelist()
    print('zip names:', names)
    media = [n for n in names if n.startswith('word/media/')]
    print('media files:', media)
    if media:
        data = zf.read(media[0])
        print('media bytes len', len(data), 'png header', data[:4])
        filt = ImageFilterPipeline(min_image_size=10, min_image_size_minor=5)
        prepared = filt.prepare_image(image_bytes=data, ext='png', width=300, height=300, image_id='t', page_number=1, img_idx=0)
        print('prepared:', prepared)
        print('filter_stats:', filt.filter_stats)
