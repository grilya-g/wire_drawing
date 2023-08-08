def coords_to_sdxf_1(r,x,k):
    #k -коэффициент для калибровочного участка
    name = input()
    doc = ezdxf.new('R2010')  # create a new DXF R2010 drawing, official DXF version name: 'AC1024'

    die_profile =[]
    for i in (range(0,len(r))):
        die_profile.append(tuple([r[i],x[i]])) #
        
    die_profile.append(tuple([min(die_profile)[0],min(die_profile)[1]-k*2*min(die_profile)[0]])) #min(die_profile)[1]
    die_profile.append(tuple([min(die_profile)[0]+min(die_profile)[0]/5,2*(-min(die_profile)[0])-k*min(die_profile)[0]])) 
    msp = doc.modelspace()  # add new entities to the modelspace
    for i in range(0,len(die_profile)-1):
        msp.add_line(die_profile[i], die_profile[i+1])  # add a LINE entity
        
    doc.saveas(str(name)+'.dxf') 
	
	
	
	
x = transform_for_scetch(twin_elleptical_die(0.771/2,0.738/2,0.1569870134946728)[1],twin_elleptical_die(0.771/2,0.738/2,0.1569870134946728)[0])[0]
y = transform_for_scetch(twin_elleptical_die(0.771/2,0.738/2,0.1569870134946728)[1],twin_elleptical_die(0.771/2,0.738/2,0.1569870134946728)[0])[1]
coords_to_sdxf_1(x,y,0.3)