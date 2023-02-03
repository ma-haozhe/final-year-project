import os

#NOTE: it doesnt work with x trailing yet!

def rename_photos(directory):
    for filename in os.listdir(directory):
        print("file name:", filename)
        if filename.endswith('.jpg'):
            count = 0
            add_counter=0
            head=""
            for char in reversed(filename):
                add_counter+=1
                if char == '0':
                    count += 1
                if add_counter>4 and add_counter<=len(filename):
                    head = (head+char).replace("x","")
                else:
                    continue
            new_filename = head+'-'+str(count*5) + '.jpg'
            os.rename(os.path.join(directory, filename),
                      os.path.join(directory, new_filename))



rename_photos('smalltest/10-15')