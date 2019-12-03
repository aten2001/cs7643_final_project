import imageio


filename_base = "ground_truthsaved_"
filename_base = "test_mesaved_"

with imageio.get_writer('movie.gif', mode='I') as writer:
    for f in range(100):

    	try:
    		file = filename_base + "{:04}.jpg".format(f)
    		print file
        	image = imageio.imread(file)
        	writer.append_data(image)

    	except:
    		break

