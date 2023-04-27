from bs4 import BeautifulSoup

with open('GaussianProcesses.html') as f:
    document = BeautifulSoup("\n".join(f.readlines()), "lxml")

    #save all the styles
    styles = document.find_all('style')
    with open('html/styles.css', 'w') as style_file:
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
            
            if (comm.contents[0][:5] == "#HIDE"):            
                remove = True
                break

        if remove:            
            c.decompose()


    #save cleaned document
    with open("html/notebook.html", "w") as main_file:
        main_file.writelines(str(document))




