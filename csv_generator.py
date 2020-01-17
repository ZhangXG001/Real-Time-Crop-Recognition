#coding:utf-8
import os
import csv    
def create_csv(dirname):  
    """
    This function creats .csv file of dataset
    Args:
    dirname: the file name of dataset
    Returns:
    a .csv file of dataset
    """
    #the path of the dataset
    path = './dataset/'+ dirname +'/'
    name = os.listdir(path)
    #sorts the data with name
    name.sort(key=lambda x: (x.split('_')[0][-1:]))
    with open (dirname+'.csv','w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        #write the name of flies in the dataset into the .csv file
        for n in name:
            if n[-4:] == '.jpg':    
                print(n)    
                writer.writerow(['./dataset/'+str(dirname) +'/'+ str(n),'./dataset/' + str(dirname) + 'label/' + str(n[:-4] + '.png')])  
            else:
                pass

if __name__ == "__main__":  
    #gets the .csv file of train and validation dataset
    create_csv('train')   
    create_csv('validation')    
