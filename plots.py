import seaborn as sns
import matplotlib.pyplot as plt



def boxhist(df,x):
    
    sns.set_theme(palette="Blues_r")

    # creating a figure composed of two matplotlib.Axes objects (ax_box and ax_hist)
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
 
    # assigning a graph to each ax
    ax = sns.boxplot(df[x], orient="h", ax=ax_box)
    sns.histplot(data=df, x=x, ax=ax_hist)
    
    # Remove x axis name for the boxplot
    ax.set(xlabel='')

    plt.show()


def bplts(df,x,y,w,h,xrot,view,lim):
    
    '''
    Function to plot bar graph for 2 variables
        df: df name
        x: x variable
        y: y variable
        w: plot width
        h: plot height
        xrot: xtick rotation
        order: order graph
        lim: limit x values
    '''
    # setting the dimensions of the plot
    fig, ax = plt.subplots(figsize=(w, h))
    sns.barplot(df, x=x, y=y, ax=ax, errorbar=None,color='#b3cede',order=orders)

    if  view == 'lim':
        orders = df[y].value_counts(ascending=False).head(lim).index
    elif view == 'asc':
        orders == df[y].value_counts(ascending=True).index
    elif  view == 'desc':
        orders == df[y].value_counts(ascending=False).index
    else:
        orders=None

    plt.title(f'{y} by {x}', fontsize=12)
    plt.ylabel(y)
    plt.xlabel(x)
    plt.xticks(rotation=xrot, ha='right')
    plt.show()


def bhplt(df,feat,w,h,wspace,bw): 
    '''
    Define function to create box plot and histograms 
        df: df name
        feat: feature name
        w: plot width
        h: plot height
        wspace: space between subplots
        bw: histogram's bin width
    '''    
    
    #Print column name
    print('Feature:', feat)  

    #Print Skew amount
    skew =  round(df[feat].skew(), 2)
    if skew > 0.5:
        print('Right Skew:',skew)
    elif skew < -0.5:
        print('Left Skew:',skew)
    else:
        print('Normal:',skew)

    # condition statement to print either median or mean
    if skew > 0.5:                          #print median
        median = df[feat].median()
        print('Median:', df[feat].median())
        line = median
    elif skew <- 0.5:
        median = df[feat].median()           #print median
        print('Median:', df[feat].median()) 
        line = median   
    else:
        mean = df[feat].mean()               #Print mean
        print('Mean:', df[feat].mean())
        line = mean

    # Set figure size
    plt.figure(figsize = (w, h))

    # Create a boxplot to visualize distribution and detect any outliers
    plt.subplot(1, 2, 1)              
    plt.title('Boxplot')
    sns.boxplot(x=df[feat],color='#b3cede')
    plt.subplots_adjust(wspace=wspace)

    # plot Histogram
    plt.subplot(1, 2, 2)   
    sns.histplot(x=df[feat],binwidth=bw,color='#b3cede', edgecolor='#78a9c8')
    plt.title('Histogram')
    plt.show()


def ctplt(df, x:str, w, h, lim,xrot, lbl:str, lbl_type:str):
    '''
    Function to plot count graph
        df: df name
        x: x variable
        w: plot width
        h: plot height
        lim: number of x values to display 
        xrot: xtick rotation degree
        lbl: display bar labels ('y','n')
        lbl_type: display as count or percentage ('%','#')
    '''

    fig = plt.figure(figsize=(w,h))
    ax = sns.countplot(x=df[x],order=df[x].value_counts(ascending=False).head(lim).index, color='#b3cede')
    plt.xticks(rotation=xrot)
    ax.tick_params(left=False, bottom=False) # Remove ticks
    sns.despine(left=True, bottom=True) # Remove borders borders

    if lbl == 'y':
        for patch in ax.patches:
            count = patch.get_height()
            percentage = (count / len(df)) * 100
            if lbl_type == '%':
                label = f"{percentage:.1f}%"  # Format as percentage with one decimal
            elif lbl_type == '#':
                label = f"{count:.0f}"  # Format as percentage with one decimal
            else:
                label=None   
            x = patch.get_x() + patch.get_width() / 2 # Place the label at the top center of the bar
            y = patch.get_height() + 0.1  # Adjust y position for better placement
            ax.text(x, y, label, ha='center', va='bottom')
    else:
        label = None
