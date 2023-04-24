from bs4 import BeautifulSoup

filename = 'tenure-effects/notebook.html'
output_folder = 'tenure-effects/html'
removal_mode = 'all'

with open(filename) as f:
    document = BeautifulSoup("\n".join(f.readlines()), "lxml")

    #save all the styles
    styles = document.find_all('style')
    with open(output_folder + '/styles.css', 'w') as style_file:
        for s in styles:            
            style_file.writelines(s.contents)
            s.decompose()

    #add styles reference to the main document
    style_ref = document.new_tag("link", href="styles.css", rel="stylesheet")
    document.find("head").append(style_ref)
    
    #remove warning outputs
    for w in document.select('div[data-mime-type="application/vnd.jupyter.stderr"]'):
        w.decompose()

    #remove code cells to be hidden
    for c in document.select(".jp-Cell-inputWrapper"):        
            remove = False
            for comm in c.select(".c1"):  
                if removal_mode == 'all':
                    remove = True
                    break
                elif (comm.contents[0][:5] == "#HIDE"):            
                    remove = True
                    break

            if remove:            
                c.decompose()


    #save cleaned document
    with open(output_folder + "/tenure-effects.html", "w") as main_file:
        main_file.writelines(str(document))
