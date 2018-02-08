//------------------------------------------------------------------------------
// ssgui: a Java GUI interface for viewing, selecting, and downloading matrices
// from the SuiteSparse Matrix Collection.  To compile this program, type the
// following in your OS command window:
//
//      javac ssgui.java
//      jar cfe ssgui.jar ssgui *.class sshelp.html
//
// You can then delete the *.class files.  To run the program, type:
// 
//      java -jar ssgui.jar
//
// In all platforms except Windows (Mac, Linux, Solaris, ...) compile with:
//
//      make
//
// and run with
//
//      make run
//
//------------------------------------------------------------------------------
// Changing the default parameters for ssgui:
//------------------------------------------------------------------------------
//
// The following parameters of ssgui can be changed by editing this file and
// recompiling (see the comments "default settings" below in the code).
// Each setting is a string:
//
//      sssite: URL for the SuiteSparse Matrix Collection
//      default is sssite = "https://sparse.tamu.edu" ;
//
//      ssarchive: directory containing your copy of the collection.
//      If blank, then it defaults to the directory containing ssgui.
//
//      refresh: refresh time, in days, for updating the index.  use INF to
//      never refresh
//
//      proxy_server: HTTP proxy server. If none (default), then leave blank.
//
//      proxy port: default is 80 if left blank
//
// Copyright (c) 2009-2017, Timothy A. Davis.  See sshelp.html for the license,
// and for help on how to use this program, or click "Help" in the GUI.
//------------------------------------------------------------------------------

import java.io.* ;
import java.util.* ;
import java.text.* ;
import java.net.* ;
import javax.swing.* ;
import javax.swing.table.* ;
import javax.swing.event.* ;
import java.awt.* ;
import java.awt.event.* ;

public class ssgui extends JFrame
{

    //--------------------------------------------------------------------------
    // private variables, accessible to all methods in this class
    //--------------------------------------------------------------------------

    private static final String
        ssstats = "files/ssstats.csv",
        ssindex = "files/ss_index.mat",
        all_kinds = "(all kinds)", all_groups = "(all groups)" ;
    private static final int K = 1024, M = K*K, buffersize = K,
        MSEC_PER_DAY = 86400000 ;

    private static long INF = Long.MAX_VALUE ;

    private long refresh ;
    private int nselected ;
    private int [ ] download_ids = null ;
    private boolean gui_ready, downloading, cancel, get_icons ;
    private boolean debug = false ;

    private matrix_Table_Model matrix_model = null ;

    private File mat, MM, RB, iconDir ;
    private String [ ] Kinds, Groups ;
    private Object [ ][ ] Stats ;

    private Date today, last_download ;

    // files and input/output streams
    private static String ftemp_name = null ;
    private static BufferedOutputStream ftemp_out = null ;
    private static BufferedInputStream url_in = null ;
    private static BufferedReader in_reader = null ;
    private static PrintWriter print_out = null ;
    private static String sssite, ssarchive, proxy_server, proxy_port ;

    // Java Swing components available to all methods in this class:
    private JTable matrix_Table ;
    private JButton download_Button, cancel_Button ;
    private JTextField minrow_Field, maxrow_Field, mincol_Field, maxcol_Field,
        minnz_Field, maxnz_Field, minpsym_Field, maxpsym_Field, minnsym_Field,
        maxnsym_Field ;
    private JRadioButton posdef_yes_Button, posdef_no_Button,
        posdef_either_Button, nd_yes_Button, nd_no_Button, nd_either_Button,
        real_yes_Button, real_no_Button, real_either_Button,
        shape_square_Button, shape_rect_Button, shape_either_Button ;
    private JLabel nselected_Label, progress_size_Label, icon_Label ;
    private JCheckBox format_mat_Button, format_mm_Button, format_rb_Button ;
    private JProgressBar progress1_Bar, progress2_Bar ;
    private JFileChooser chooser ;
    private JList Group_List, Kind_List ;

    //--------------------------------------------------------------------------
    // create the GUI
    //--------------------------------------------------------------------------

    private ssgui ( )
    {
        gui_ready = false ;
        downloading = false ;
        cancel = false ;
        final Font plain_Font = new Font ("SansSerif", Font.PLAIN, 12) ;
        final Font small_Font = new Font ("SansSerif", Font.PLAIN, 10) ;
        today = new Date ( ) ;
        last_download = new Date ( ) ;

        //----------------------------------------------------------------------
        // default settings.  Edit this file and recompile to change them.
        //----------------------------------------------------------------------

        // If ssarchive is blank, then it defaults to the current directory.
        ssarchive = "" ;

        // URL for the SuiteSparse Matrix Collection
        sssite = "https://sparse.tamu.edu" ;

        // refresh time, in days.  use INF to never refresh
        refresh = 30 ;   

        // HTTP proxy server. If none (default), then leave blank.
        proxy_server = "" ;

        // proxy port (default is 80 if left blank)
        proxy_port = "" ;

        //----------------------------------------------------------------------
        //----------------------------------------------------------------------

        // set up the HTTP proxy
        if (proxy_server.length ( ) > 0)
        {
            if (proxy_port.length ( ) == 0)
            {
                proxy_port = "80" ;
            }
            // set the proxy server and port
            System.setProperty ("proxySet", "true" ) ;
            System.setProperty ("http.proxyHost", proxy_server) ;
            System.setProperty ("http.proxyPort", proxy_port) ;
        }

        // ssarchive defaults to current working directory, if empty
        if (ssarchive.length ( ) == 0)
        {
            ssarchive = System.getProperty ("user.dir") ;
        }
        ssarchive = ssarchive.replace ('\\', File.separatorChar) ;
        ssarchive = ssarchive.replace ('/', File.separatorChar) ;
        char c = ssarchive.charAt (ssarchive.length ( ) - 1) ;
        if (c != File.separatorChar)
        {
            ssarchive += File.separatorChar ;
        }

        if (debug)
        {
            System.out.println ("") ;
            System.out.println ("ssgui, debugging enabled.") ;
            System.out.println ("local archive: [" + ssarchive    + "]") ;
            System.out.println ("ss url:        [" + sssite       + "]") ;
            System.out.println ("refresh:       [" + refresh      + "]") ;
            System.out.println ("proxy server:  [" + proxy_server + "]") ;
            System.out.println ("proxy port:    [" + proxy_port   + "]") ;
        }

        //----------------------------------------------------------------------
        // make sure the top-level directories exist

        mat = CheckDir ("mat") ;
        MM = CheckDir ("MM") ;
        RB = CheckDir ("RB") ;
        iconDir = CheckDir ("files") ;

        //----------------------------------------------------------------------
        // read in the matrix statistics

        Stats = load_ssstats ( ) ;

        if (Stats == null ||
            ((today.getTime ( ) - last_download.getTime ( )) / MSEC_PER_DAY
            > refresh))
        {
            // ssstats file is missing, or old.  Download both
            // files/ssstats.csv and mat/ss_index.mat.
            Stats = download_matrix_stats ( ) ;
            if (debug) System.out.println ("downloading new ssstats.csv file") ;
        }

        if (Stats == null)
        {
            // display error dialog and quit
            JOptionPane.showMessageDialog (this,
                "Download of matrix statistics file failed.",
                "Error", JOptionPane.ERROR_MESSAGE) ;
            System.exit (-1) ;
        }

        //----------------------------------------------------------------------
        // set the title, and close on [x]

        setTitle ("ssgui: SuiteSparse Matrix Collection") ;
        setDefaultCloseOperation (WindowConstants.EXIT_ON_CLOSE) ;

        //----------------------------------------------------------------------
        // selection buttons

        JPanel select_Button_Panel = new JPanel ( ) ;

        JButton select_Button = new JButton ("Select") ;
        JButton unselect_Button = new JButton ("Deselect") ;
        JButton reset_Button = new JButton ("Reset criteria") ;
        JButton clear_Button = new JButton ("Clear selections") ;
        JButton help_Button = new JButton ("Help") ;

        select_Button_Panel.add (select_Button) ;
        select_Button_Panel.add (unselect_Button) ;
        select_Button_Panel.add (reset_Button) ;
        select_Button_Panel.add (clear_Button) ;
        select_Button_Panel.add (help_Button) ;

        select_Button.setToolTipText
        ("Click to add matrices that fit the criteria to your selection.") ;
        unselect_Button.setToolTipText ("Click to remove matrices " +
            "that fit the criteria from your selection.") ;
        reset_Button.setToolTipText ("Click to reset criteria, above.  " +
            "Prior selections, below, are not cleared.") ;
        clear_Button.setToolTipText ("Click to clear selections, below.  " +
            "Criteria, above, is not reset).") ;
        help_Button.setToolTipText ("For help, click here") ;

        select_Button.addActionListener
        (
            new ActionListener ( )
            {
                public void actionPerformed (ActionEvent e)
                {
                    make_selection (true) ;
                }
            }
        ) ;

        unselect_Button.addActionListener
        (
            new ActionListener ( )
            {
                public void actionPerformed (ActionEvent e)
                {
                    make_selection (false) ;
                }
            }
        ) ;

        reset_Button.addActionListener
        (
            new ActionListener ( )
            {
                public void actionPerformed (ActionEvent e)
                {
                    reset_Button_action (e) ;
                }
            }
        ) ;

        clear_Button.addActionListener
        (
            new ActionListener ( )
            {
                public void actionPerformed (ActionEvent e)
                {
                    clear_Button_action (e) ;
                }
            }
        ) ;

        help_Button.addActionListener
        (
            new ActionListener ( )
            {
                public void actionPerformed (ActionEvent e)
                {
                    help_Button_action (e) ;
                }
            }
        ) ;

        //----------------------------------------------------------------------
        // download button and format options

        JPanel format_Panel = new JPanel ( ) ;

        format_mat_Button = new JCheckBox ("MATLAB (mat)") ;
        format_mm_Button = new JCheckBox ("Matrix Market (MM)") ;
        format_rb_Button = new JCheckBox ("Rutherford/Boeing (RB)    ") ;

        format_mat_Button.setSelected (true) ;

        format_mat_Button.setToolTipText ("Download in MATLAB *.mat format.") ;
        format_mm_Button.setToolTipText ("Download in Matrix Market.") ;
        format_rb_Button.setToolTipText
            ("Download in Rutherford/Boeing format.") ;

        nselected = 0 ;
        nselected_Label = new JLabel ( ) ;
        download_Button = new JButton ("Download") ;

        format_Panel.add (download_Button) ;
        format_Panel.add (format_mat_Button) ;
        format_Panel.add (format_mm_Button) ;
        format_Panel.add (format_rb_Button) ;
        format_Panel.add (nselected_Label) ;
        format_Panel.setMaximumSize (new Dimension (0,0)) ;

        // progress bar and cancel button
        FlowLayout progress_Layout = new FlowLayout (FlowLayout.LEADING) ;
        JPanel progress_Panel = new JPanel (progress_Layout) ;

        cancel_Button = new JButton ("Cancel") ;
        cancel_Button.setEnabled (false) ;
        progress1_Bar = new JProgressBar ( ) ;
        progress2_Bar = new JProgressBar ( ) ;
        progress_size_Label = new JLabel ("") ;
        progress1_Bar.setMinimumSize (new Dimension (200,16)) ;
        progress2_Bar.setMinimumSize (new Dimension (200,16)) ;
        progress_Panel.add (cancel_Button) ;
        progress_Panel.add (new JLabel ("   Overall progress:")) ;
        progress_Panel.add (progress1_Bar) ;
        progress_Panel.add (new JLabel ("   Current file:")) ;
        progress_Panel.add (progress2_Bar) ;
        progress_Panel.add (progress_size_Label) ;
        progress_Panel.setMaximumSize (new Dimension (0,0)) ;
        cancel_Button.setToolTipText ("No downloads in progress.") ;

        download_Button.addActionListener
        (
            new ActionListener ( )
            {
                public void actionPerformed (ActionEvent e)
                {
                    download_Button_action (e) ;
                }
            }
        ) ;

        cancel_Button.addActionListener
        (
            new ActionListener ( )
            {
                public void actionPerformed (ActionEvent e)
                {
                    cancel_Button_action (e) ;
                }
            }
        ) ;

        JPanel download_Panel = new JPanel ( ) ;
        GroupLayout layout3 = new GroupLayout (download_Panel) ;
        download_Panel.setLayout (layout3) ;

        layout3.setAutoCreateGaps (true) ;
        layout3.setAutoCreateContainerGaps (false) ;

        layout3.setHorizontalGroup
        (
            layout3.createParallelGroup (GroupLayout.Alignment.LEADING)
                .addComponent (format_Panel)
                .addComponent (progress_Panel)
        ) ;

        layout3.setVerticalGroup
        (
            layout3.createSequentialGroup ( )
                .addComponent (format_Panel)
                .addComponent (progress_Panel)
        ) ;

        download_Panel.setBorder
            (BorderFactory.createTitledBorder ("download")) ;
        download_Panel.setMaximumSize (new Dimension (0,0)) ;

        //----------------------------------------------------------------------
        // panel for m, n, nnz, psym, and nsym

        // # of rows
        minrow_Field = new JTextField ("") ;
        JLabel rowlabel = new JLabel (" \u2264 number of rows \u2264 ") ;
        maxrow_Field = new JTextField ("") ;
        minrow_Field.setColumns (16) ;
        maxrow_Field.setColumns (16) ;
        minrow_Field.setToolTipText ("Leave blank for 'zero'.") ;
        maxrow_Field.setToolTipText ("Leave blank for 'infinite'.") ;
        minrow_Field.setMinimumSize (new Dimension (120,0)) ;
        maxrow_Field.setMinimumSize (new Dimension (120,0)) ;

        // # of columns
        mincol_Field = new JTextField ("") ;
        JLabel collabel = new JLabel (" \u2264 number of columns \u2264 ") ;
        maxcol_Field = new JTextField ("") ;
        mincol_Field.setColumns (16) ;
        maxcol_Field.setColumns (16) ;
        mincol_Field.setToolTipText ("Leave blank for 'zero'.") ;
        maxcol_Field.setToolTipText ("Leave blank for 'infinite'.") ;
        mincol_Field.setMinimumSize (new Dimension (120,0)) ;
        maxcol_Field.setMinimumSize (new Dimension (120,0)) ;

        // # of entries
        minnz_Field = new JTextField ("") ;
        JLabel nnzlabel = new JLabel (" \u2264 number of nonzeros \u2264 ") ;
        maxnz_Field = new JTextField ("") ;
        minnz_Field.setColumns (16) ;
        maxnz_Field.setColumns (16) ;
        minnz_Field.setToolTipText ("Leave blank for 'zero'.") ;
        maxnz_Field.setToolTipText ("Leave blank for 'infinite'.") ;
        minnz_Field.setMinimumSize (new Dimension (120,0)) ;
        maxnz_Field.setMinimumSize (new Dimension (120,0)) ;

        // pattern symmetry
        minpsym_Field = new JTextField ("0.0") ;
        JLabel psymlabel = new JLabel (" \u2264 pattern symmetry \u2264 ") ;
        maxpsym_Field = new JTextField ("1.0") ;
        minpsym_Field.setColumns (16) ;
        maxpsym_Field.setColumns (16) ;
        maxpsym_Field.setToolTipText (
        "Refers to position of nonzeros, not their values.\n" +
        "1 = perfectly symmetric pattern, 0 = perfectly unsymmetric pattern.") ;
        minpsym_Field.setMinimumSize (new Dimension (120,0)) ;
        maxpsym_Field.setMinimumSize (new Dimension (120,0)) ;

        // numerical symmetry
        minnsym_Field = new JTextField ("0.0") ;
        JLabel nsymlabel = new JLabel (" \u2264 numerical symmetry \u2264 ") ;
        maxnsym_Field = new JTextField ("1.0") ;
        minnsym_Field.setColumns (16) ;
        maxnsym_Field.setColumns (16) ;
        maxnsym_Field.setToolTipText (
        "1 means A=A', 0 means no nonzero entry A(i,j) = A(j,i).") ;
        minnsym_Field.setMinimumSize (new Dimension (120,0)) ;
        maxnsym_Field.setMinimumSize (new Dimension (120,0)) ;

        JPanel range_Panel = new JPanel ( ) ;
        GroupLayout layout5 = new GroupLayout (range_Panel) ;
        range_Panel.setLayout (layout5) ;
        layout5.setAutoCreateGaps (false) ;
        layout5.setAutoCreateContainerGaps (false) ;

        layout5.setHorizontalGroup
        (
            layout5.createSequentialGroup ( )
                .addGroup
                (
                    layout5.createParallelGroup (GroupLayout.Alignment.LEADING)
                        .addComponent (minrow_Field)
                        .addComponent (mincol_Field)
                        .addComponent (minnz_Field)
                        .addComponent (minpsym_Field)
                        .addComponent (minnsym_Field)
                )
                .addGroup
                (
                    layout5.createParallelGroup (GroupLayout.Alignment.LEADING)
                        .addComponent (rowlabel)
                        .addComponent (collabel)
                        .addComponent (nnzlabel)
                        .addComponent (psymlabel)
                        .addComponent (nsymlabel)
                )
                .addGroup
                (
                    layout5.createParallelGroup (GroupLayout.Alignment.LEADING)
                        .addComponent (maxrow_Field)
                        .addComponent (maxcol_Field)
                        .addComponent (maxnz_Field)
                        .addComponent (maxpsym_Field)
                        .addComponent (maxnsym_Field)
                )
        ) ;

        layout5.setVerticalGroup
        (
            layout5.createSequentialGroup ( )

                .addGroup
                (
                    layout5.createParallelGroup (GroupLayout.Alignment.LEADING)
                        .addComponent (minrow_Field)
                        .addComponent (rowlabel)
                        .addComponent (maxrow_Field)
                )
                .addGroup
                (
                    layout5.createParallelGroup (GroupLayout.Alignment.LEADING)
                        .addComponent (mincol_Field)
                        .addComponent (collabel)
                        .addComponent (maxcol_Field)
                )
                .addGroup
                (
                    layout5.createParallelGroup (GroupLayout.Alignment.LEADING)
                        .addComponent (minnz_Field)
                        .addComponent (nnzlabel)
                        .addComponent (maxnz_Field)
                )
                .addGroup
                (
                    layout5.createParallelGroup (GroupLayout.Alignment.LEADING)
                        .addComponent (minpsym_Field)
                        .addComponent (psymlabel)
                        .addComponent (maxpsym_Field)
                )
                .addGroup
                (
                    layout5.createParallelGroup (GroupLayout.Alignment.LEADING)
                        .addComponent (minnsym_Field)
                        .addComponent (nsymlabel)
                        .addComponent (maxnsym_Field)
                )

        ) ;

        range_Panel.setMaximumSize (new Dimension (0,0)) ;
        // range_Panel.setBorder (BorderFactory.createTitledBorder ("range")) ;

        //----------------------------------------------------------------------
        // checkbox panel for posdef, ND, real, and format

        // square or rectangular
        JLabel shape_label = new JLabel ("shape ") ;
        shape_square_Button = new JRadioButton ("square   ") ;
        shape_rect_Button = new JRadioButton ("rectangular   ") ;
        shape_either_Button = new JRadioButton ("either   ") ;
        shape_either_Button.setSelected (true) ;
        ButtonGroup shape_group = new ButtonGroup ( ) ;
        shape_group.add (shape_square_Button) ;
        shape_group.add (shape_rect_Button) ;
        shape_group.add (shape_either_Button) ;
        shape_square_Button.setToolTipText
            ("Select 'yes' for square matrices.") ;
        shape_rect_Button.setToolTipText
            ("Select 'no' for rectangular matrices only.") ;
        shape_either_Button.setToolTipText
            ("Select 'either' for any matrix.") ;

        // positive definite
        JLabel posdef_label = new JLabel ("positive definite? ") ;
        posdef_yes_Button = new JRadioButton ("yes") ;
        posdef_no_Button = new JRadioButton ("no") ;
        posdef_either_Button = new JRadioButton ("either") ;
        posdef_either_Button.setSelected (true) ;
        ButtonGroup posdef_group = new ButtonGroup ( ) ;
        posdef_group.add (posdef_yes_Button) ;
        posdef_group.add (posdef_no_Button) ;
        posdef_group.add (posdef_either_Button) ;

        posdef_yes_Button.setToolTipText
            ("Select 'yes' for symmetric positive definite matrices only.") ;
        posdef_no_Button.setToolTipText
            ("Select 'no' for non-positive definite matrices only.") ;
        posdef_either_Button.setToolTipText
            ("Select 'either' for any matrix.") ;

        // 2D/3D
        JLabel nd_label = new JLabel ("2D/3D discretization?    ") ;
        nd_yes_Button = new JRadioButton ("yes") ;
        nd_no_Button = new JRadioButton ("no") ;
        nd_either_Button = new JRadioButton ("either") ;
        nd_either_Button.setSelected (true) ;
        ButtonGroup nd_group = new ButtonGroup ( ) ;
        nd_group.add (nd_yes_Button) ;
        nd_group.add (nd_no_Button) ;
        nd_group.add (nd_either_Button) ;

        nd_yes_Button.setToolTipText
            ("Select 'yes' for matrices " +
            "arising from 2D or 3D discretizations only.") ;
        nd_no_Button.setToolTipText
            ("Select 'no' to exclude matrices " +
            "arising from 2D or 3D discretizations.") ;
        nd_either_Button.setToolTipText ("Select 'either' for any matrix.") ;

        // real or complex
        JLabel real_label = new JLabel ("real or complex? ") ;
        real_yes_Button = new JRadioButton ("real") ;
        real_no_Button = new JRadioButton ("complex") ;
        real_either_Button = new JRadioButton ("either") ;
        real_either_Button.setSelected (true) ;
        ButtonGroup real_group = new ButtonGroup ( ) ;
        real_group.add (real_yes_Button) ;
        real_group.add (real_no_Button) ;
        real_group.add (real_either_Button) ;

        real_yes_Button.setToolTipText
            ("Select 'real' for real matrices only (includes integer and binary).") ;
        real_no_Button.setToolTipText
            ("Select 'complex' for complex matrices only.") ;
        real_either_Button.setToolTipText
            ("Select 'either' for any matrix.") ;

        JPanel check_Panel = new JPanel ( ) ;
        GroupLayout layout4 = new GroupLayout (check_Panel) ;
        check_Panel.setLayout (layout4) ;
        layout4.setAutoCreateGaps (false) ;
        layout4.setAutoCreateContainerGaps (false) ;

        layout4.setHorizontalGroup
        (
            layout4.createSequentialGroup ( )
                .addGroup
                (
                    layout4.createParallelGroup (GroupLayout.Alignment.LEADING)
                        .addComponent (shape_label)
                        .addComponent (posdef_label)
                        .addComponent (nd_label)
                        .addComponent (real_label)
                )
                .addGroup
                (
                    layout4.createParallelGroup (GroupLayout.Alignment.LEADING)
                        .addComponent (shape_square_Button)
                        .addComponent (posdef_yes_Button)
                        .addComponent (nd_yes_Button)
                        .addComponent (real_yes_Button)
                )
                .addGroup
                (
                    layout4.createParallelGroup (GroupLayout.Alignment.LEADING)
                        .addComponent (shape_rect_Button)
                        .addComponent (posdef_no_Button)
                        .addComponent (nd_no_Button)
                        .addComponent (real_no_Button)
                )
                .addGroup
                (
                    layout4.createParallelGroup (GroupLayout.Alignment.LEADING)
                        .addComponent (shape_either_Button)
                        .addComponent (posdef_either_Button)
                        .addComponent (nd_either_Button)
                        .addComponent (real_either_Button)
                )
        ) ;

        layout4.setVerticalGroup
        (
            layout4.createSequentialGroup ( )
                .addGroup
                (
                    layout4.createParallelGroup (GroupLayout.Alignment.LEADING)
                        .addComponent (shape_label)
                        .addComponent (shape_square_Button)
                        .addComponent (shape_rect_Button)
                        .addComponent (shape_either_Button)
                )
                .addGroup
                (
                    layout4.createParallelGroup (GroupLayout.Alignment.LEADING)
                        .addComponent (posdef_label)
                        .addComponent (posdef_yes_Button)
                        .addComponent (posdef_no_Button)
                        .addComponent (posdef_either_Button)
                )
                .addGroup
                (
                    layout4.createParallelGroup (GroupLayout.Alignment.LEADING)
                        .addComponent (nd_label)
                        .addComponent (nd_yes_Button)
                        .addComponent (nd_no_Button)
                        .addComponent (nd_either_Button)
                )
                .addGroup
                (
                    layout4.createParallelGroup (GroupLayout.Alignment.LEADING)
                        .addComponent (real_label)
                        .addComponent (real_yes_Button)
                        .addComponent (real_no_Button)
                        .addComponent (real_either_Button)
                )
        ) ;

        check_Panel.setMaximumSize (new Dimension (0,0)) ;

        //----------------------------------------------------------------------
        // Group and Kind lists

        Kinds = FindKinds ( ) ;
        Groups = FindGroups ( ) ;

        // Group_List = new JList ((Object [ ]) Groups) ;
        // Kind_List = new JList ((Object [ ]) Kinds) ;
        Group_List = new JList<String> (Groups) ;
        Kind_List = new JList<String> (Kinds) ;

        JScrollPane Group_Pane = new JScrollPane (Group_List) ;
        JScrollPane Kind_Pane = new JScrollPane (Kind_List) ;

        Kind_Pane.setBorder (BorderFactory.createTitledBorder ("kind")) ;
        Group_Pane.setBorder (BorderFactory.createTitledBorder ("group")) ;

        Group_List.setFont (plain_Font) ;
        Kind_List.setFont (plain_Font) ;

        Group_Pane.setVerticalScrollBarPolicy
            (ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS) ;
        Kind_Pane.setVerticalScrollBarPolicy
            (ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS) ;

        Group_List.setVisibleRowCount (5) ;
        Kind_List.setVisibleRowCount (5) ;

        JPanel list_Panel = new JPanel ( ) ;
        GroupLayout layout9 = new GroupLayout (list_Panel) ;
        list_Panel.setLayout (layout9) ;

        layout9.setAutoCreateGaps (true) ;
        layout9.setAutoCreateContainerGaps (false) ;

        layout9.setHorizontalGroup
        (
            layout9.createSequentialGroup ( )
                .addComponent (Group_Pane)
                .addComponent (Kind_Pane)
        ) ;

        layout9.setVerticalGroup
        (
            layout9.createParallelGroup (GroupLayout.Alignment.LEADING)
                .addComponent (Group_Pane)
                .addComponent (Kind_Pane)
        ) ;

        list_Panel.setMinimumSize (new Dimension (450,150)) ;

        //----------------------------------------------------------------------
        // selection panel
        JPanel selection_Panel = new JPanel ( ) ;
        GroupLayout layout2 = new GroupLayout (selection_Panel) ;
        selection_Panel.setLayout (layout2) ;
        layout2.setAutoCreateGaps (true) ;
        layout2.setAutoCreateContainerGaps (false) ;
        layout2.setHorizontalGroup
        (
            layout2.createParallelGroup (GroupLayout.Alignment.LEADING)
                .addComponent (range_Panel)
                .addComponent (check_Panel)
                .addComponent (list_Panel)
                .addComponent (select_Button_Panel)
        ) ;
        layout2.setVerticalGroup
        (
            layout2.createSequentialGroup ( )
                .addComponent (range_Panel)
                .addComponent (check_Panel)
                .addComponent (list_Panel)
                .addComponent (select_Button_Panel)
        ) ;
        selection_Panel.setBorder
            (BorderFactory.createTitledBorder ("selection criteria")) ;
        selection_Panel.setMaximumSize (new Dimension (0,0)) ;

        //----------------------------------------------------------------------
        // create the table of matrices

        matrix_model = new matrix_Table_Model ( ) ;
        matrix_Table = new JTable (matrix_model)
        {
            // table header tool tips
            protected JTableHeader createDefaultTableHeader ( )
            {
                return new JTableHeader (columnModel)
                {
                    public String getToolTipText (MouseEvent e)
                    {
                        String tip = null ;
                        java.awt.Point p = e.getPoint ( ) ;
                        int i = columnModel.getColumnIndexAtX (p.x) ;
                        int j = columnModel.getColumn (i).getModelIndex ( ) ;
                        return matrix_column_tooltips [j] ;
                    }
                } ;
            }
        } ;

        JTableHeader header = matrix_Table.getTableHeader ( ) ;
        final TableCellRenderer hr = header.getDefaultRenderer ( ) ;
        header.setDefaultRenderer
        (
            new TableCellRenderer ( )
            {
                public Component getTableCellRendererComponent (JTable table,
                    Object value, boolean isSelected, boolean hasFocus,
                    int row, int column)
                {
                    Component co = hr.getTableCellRendererComponent (
                        table, value, isSelected, hasFocus, row, column) ;
                    co.setFont (small_Font) ;
                    return (co) ;
                }
            }
        ) ;

        matrix_model.load_data (Stats) ;

        //----------------------------------------------------------------------
        // popup menu for the table

        JPopupMenu popup = new JPopupMenu ( ) ;

        JMenuItem select_menuItem =
            new JMenuItem ("Select highlighted matrices") ;
        select_menuItem.addActionListener
        (
            new ActionListener ( )
            {
                public void actionPerformed (ActionEvent e)
                {
                    popup_action (e, true) ;
                }
            }
        ) ;

        JMenuItem unselect_menuItem =
            new JMenuItem ("Deselect highlighted matrices") ;
        unselect_menuItem.addActionListener
        (
            new ActionListener ( )
            {
                public void actionPerformed (ActionEvent e)
                {
                    popup_action (e, false) ;
                }
            }
        ) ;

        JMenuItem exportcsv_menuItem =
            new JMenuItem ("Export selected matrices as CSV file") ;
        exportcsv_menuItem.addActionListener
        (
            new ActionListener ( )
            {
                public void actionPerformed (ActionEvent e)
                {
                    export_list_action (e, true) ;
                }
            }
        ) ;

        JMenuItem exportm_menuItem =
            new JMenuItem ("Export selected matrices as MATLAB *.m file") ;
        exportm_menuItem.addActionListener
        (
            new ActionListener ( )
            {
                public void actionPerformed (ActionEvent e)
                {
                    export_list_action (e, false) ;
                }
            }
        ) ;

        popup.add (select_menuItem) ;
        popup.add (unselect_menuItem) ;
        popup.add (exportcsv_menuItem) ;
        popup.add (exportm_menuItem) ;

        // Add listener to components that can bring up popup menus.
        matrix_Table.addMouseListener (new matrix_Table_PopupListener (popup)) ;

        //----------------------------------------------------------------------
        // set default column widths

        int [ ] columnwidth = {
            40,     // 0:select
            30,     // 1:mat
            25,     // 2:MM
            25,     // 3:RB
            38,     // 4:id
            110,    // 5:Group
            150,    // 6:Name
            70,     // 7:nrows
            70,     // 8:ncols
            70,     // 9:nnz
            40,     // 10:isReal
            40,     // 11:isBinary
            40,     // 12:isND
            40,     // 13:posdef
            50,     // 14:psym
            50,     // 15:nsym
            200 } ; // 16:kind

        TableColumn column = null ;
        for (int col = 0 ; col < 17 ; col++)
        {
            column = matrix_Table.getColumnModel ( ).getColumn (col) ;
            column.setPreferredWidth (columnwidth [col]) ;
        }

        //----------------------------------------------------------------------
        // set the view size, sort by id, and add the table to a scroll pane

        matrix_Table.setPreferredScrollableViewportSize
            (new Dimension (500,70)) ;
        matrix_Table.setFillsViewportHeight (true) ;
        matrix_Table.setAutoCreateRowSorter (true) ;

        matrix_Table.getSelectionModel ( )
            .addListSelectionListener (new matrix_Table_RowListener ( )) ;

        // sort by id
        java.util.List <RowSorter.SortKey> sortKeys =
            new ArrayList<RowSorter.SortKey> ( ) ;
        sortKeys.add (new RowSorter.SortKey (4, SortOrder.ASCENDING)) ;
        (matrix_Table.getRowSorter ( )).setSortKeys (sortKeys) ; 

        matrix_Table.getTableHeader ( ).setReorderingAllowed (false) ;
        JScrollPane scroll_Pane = new JScrollPane (matrix_Table) ;
        scroll_Pane.setBorder (BorderFactory.createTitledBorder (ssarchive)) ;

        //----------------------------------------------------------------------
        // create the icon and display the default matrix

        icon_Label = new JLabel ( ) ;
        icon_Label.setFont (plain_Font) ;
        icon_Label.setVerticalTextPosition (JLabel.BOTTOM) ;
        icon_Label.setHorizontalTextPosition (JLabel.CENTER) ;
        icon_Label.setBorder (BorderFactory.createTitledBorder ("matrix icon"));
        update_icon ("HB/west0479") ;

        //----------------------------------------------------------------------
        // create the top panel (selection panel and icon)

        JPanel top_Panel = new JPanel ( ) ;
        GroupLayout layout8 = new GroupLayout (top_Panel) ;
        top_Panel.setLayout (layout8) ;

        layout8.setAutoCreateGaps (true) ;
        layout8.setAutoCreateContainerGaps (false) ;

        layout8.setHorizontalGroup
        (
            layout8.createSequentialGroup ( )
                .addComponent (selection_Panel)
                .addComponent (icon_Label)
        ) ;

        layout8.setVerticalGroup
        (
            layout8.createParallelGroup (GroupLayout.Alignment.LEADING)
                .addComponent (selection_Panel)
                .addComponent (icon_Label)
        ) ;

        top_Panel.setMaximumSize (new Dimension (0,0)) ;

        //----------------------------------------------------------------------
        // create the root layout manager

        Container pane = getContentPane ( ) ;
        GroupLayout layout = new GroupLayout (pane) ;
        pane.setLayout (layout) ;
        layout.setAutoCreateGaps (true) ;
        layout.setAutoCreateContainerGaps (false) ;

        layout.setHorizontalGroup
        (
            layout.createParallelGroup (GroupLayout.Alignment.LEADING)
                .addComponent (top_Panel)
                .addComponent (scroll_Pane)
                .addComponent (download_Panel)
        ) ;

        layout.setVerticalGroup
        (
            layout.createSequentialGroup ( )
                .addComponent (top_Panel)
                .addComponent (scroll_Pane)
                .addComponent (download_Panel)
        ) ;

        setSize (1100,750) ;

        //----------------------------------------------------------------------
        // create the file chooser; not shown until "export" is chosen

        chooser = new JFileChooser ( ) ;
        chooser.setFileSelectionMode (JFileChooser.FILES_AND_DIRECTORIES) ;

        gui_ready = true ;
        set_selected_label (true) ;

        //----------------------------------------------------------------------
        // start a thread to download any icons not present

        get_all_icons ( ) ;
    }

    //--------------------------------------------------------------------------
    // yes/no/unknown
    //--------------------------------------------------------------------------

    private String yes_no (int k)
    {
        if (k < 0)
        {
            return ("?") ;
        }
        else if (k == 0)
        {
            return ("no") ;
        }
        else
        {
            return ("yes") ;
        }
    }

    //--------------------------------------------------------------------------
    // ternary
    //--------------------------------------------------------------------------

    private int ternary (String s)
    {
        long k = Long.parseLong (s) ;
        if (k < 0)
        {
            return (-1) ;
        }
        else if (k == 0)
        {
            return (0) ;
        }
        else
        {
            return (1) ;
        }
    }

    //--------------------------------------------------------------------------
    // read the ssstats file
    //--------------------------------------------------------------------------

    private Object [ ][ ] load_ssstats ( )
    {
        if (debug) System.out.println ("reading ssstats.csv file") ;
        Object [ ][ ] S = null ;
        int nmatrices = 0 ;
        in_reader = null ;
        try
        {
            // get the number of matrices in the ss Sparse Matrix Collection
            in_reader = new BufferedReader (new FileReader
                (fix_name (ssstats))) ;
            nmatrices = Integer.parseInt (in_reader.readLine ( )) ;
            // skip past the creation date and time
            String ignore = in_reader.readLine ( ) ;
            // get the time of last download from the file modification time
            last_download =
                new Date (new File (fix_name (ssstats)).lastModified ( )) ;
        }
        catch (Exception e)
        {
            // this is OK, for now, since we can try to download a new one
            if (debug) System.out.println ("reading ssstats.csv file failed") ;
            return (null) ;
        }
        try
        {
            // read the rest of the file
            S = new Object [nmatrices][13] ;
            for (int id = 1 ; id <= nmatrices ; id++)
            {
                // split the tokens by comma
                String [ ] r = (in_reader.readLine ( )).split (",") ;
                S [id-1][0]  = id ;                             // id
                S [id-1][1]  = r [0] ;                          // Group
                S [id-1][2]  = r [1] ;                          // Name
                S [id-1][3]  = Long.parseLong (r [2]) ;         // nrows
                S [id-1][4]  = Long.parseLong (r [3]) ;         // ncols
                S [id-1][5]  = Long.parseLong (r [4]) ;         // nnz

                S [id-1][6]  = ternary (r [5]) ;                // isReal
                S [id-1][7]  = ternary (r [6]) ;                // isBinary
                S [id-1][8]  = ternary (r [7]) ;                // isND
                S [id-1][9]  = ternary (r [8]) ;                // posdef

                S [id-1][10] = Double.parseDouble (r [9]) ;     // psym
                S [id-1][11] = Double.parseDouble (r [10]) ;    // nsym
                S [id-1][12] = r [11] ;                         // kind
            }
        }
        catch (Exception e)
        {
            // this is OK, for now, since we can try to download a new one
            if (debug) System.out.println ("reading ssstats.csv file failed") ;
            return (null) ;
        }
        finally
        {
            close_reader (in_reader) ;
        }
        return (S) ;
    }

    //--------------------------------------------------------------------------
    // tool tips for each column of the matrix table
    //--------------------------------------------------------------------------

    protected String [ ] matrix_column_tooltips =
    {
        // 0:select:
        "Click to select a matrix.  This is the only column you can edit.",
        "'x' if MAT format already downloaded",                 // 1:mat
        "'x' if MM format already downloaded",                  // 2:MM
        "'x' if RB format already downloaded",                  // 3:MM
        "matrix id",                                        // 4:id
        "matrix group (typically a person or organization)",// 5:Group
        "matrix name (full name is Group/Name)",            // 6:Name
        "# of rows in the matrix",                          // 7:nrows
        "# of columns in the matrix",                       // 8:ncols
        "# of nonzeros in the matrix",                      // 9:nnz
        "if the matrix is real (not complex)",              // 10:isReal
        "if the matrix is binary",                          // 11:isBinary
        "if the matrix arises from a 2D/3D discretization", // 12:isND
        "if the matrix is symmetric positive definite",     // 13:posdef
        // 14:psym:
        "symmetry of nonzero pattern (0: none, 1: pattern(A)=pattern(A')",
        "symmetry of nonzero values (0: none, 1: A=A'",     // 15:nsym
        // 16:kind:
        "the matrix 'kind' is the problem domain from which it arises" 
    } ;

    //--------------------------------------------------------------------------
    // control whether changes to the table cause updates to fire
    //--------------------------------------------------------------------------

    public boolean fire_status = true ;

    public void fire_updates (boolean fire)
    {
        fire_status = fire ;
        if (fire)
        {
            // touch the table to force a fire
            set_table_value (get_table_value (1, 0), 1, 0) ;
        }
    }

    //--------------------------------------------------------------------------
    // table of matrix statistics
    //--------------------------------------------------------------------------

    class matrix_Table_Model extends AbstractTableModel
    {
        private String [ ] columnNames =
            {
            "select", "mat", "MM", "RB",
            "id", "Group", "Name", "# rows", "# cols", "# nonzeros", "real",
            "binary", "2D/3D", "posdef", "psym", "nsym", "kind" } ;

        private Object [ ][ ] data = null ;

        public int getColumnCount ( )
        {
            return (columnNames.length) ;
        }

        public int getRowCount ( )
        {
            return (data.length) ;
        }

        public String getColumnName (int col)
        {
            return (columnNames [col]) ;
        }

        public Object getValueAt (int row, int col)
        {
            return (data [row][col]) ;
        }

        public boolean isCellEditable (int row, int col)
        {
            // only the "select" column is edittable
            return (col == 0) ;
        }

        public void setValueAt (Object value, int row, int col)
        {
            if (col == 0 && gui_ready && ((Boolean) data [row][0]) != value)
            {
                if ((Boolean) value == false)
                {
                    // changing from selected to unselected
                    nselected-- ;
                }
                else
                {
                    // changing from unselected to selected
                    nselected++ ;
                }
                set_selected_label (download_Button.isEnabled ( )) ;
            }
            data [row][col] = value ;
            if (fire_status) fireTableDataChanged ( ) ;
        }

        public Class getColumnClass (int col)
        {
            return (getValueAt (0, col).getClass ( )) ;
        }

        public void load_data (Object [ ][ ] newstats)
        {
            // load the matrix table with all matrix statistics
            data = new Object [newstats.length][17] ;
            nselected = 0 ;
            for (int i = 0 ; i < newstats.length ; i++)
            {
                // i and j are in terms of the view, but the table is not yet
                // sorted because it is not yet visible
                data [i][0] = false ;   // select column is false
                for (int j = 1 ; j < 4 ; j++)
                {
                    // mat, MM, and RB, which can change later:
                    data [i][j] = "-" ;
                }
                for (int j = 0 ; j < 13 ; j++)
                {
                    // matrix stats, which do not change:
                    // 4:id, 5:Group, 6:Name, 7:nrows, 8:ncols, 9:nnz,
                    // 10:isreal, 11:isBinary, 12:isND, 13:posdef, 14: psym,
                    // 15:nsym, 16:kind
                    if (j >= 6 && j <= 9)
                    {
                        int k = (Integer) newstats [i][j] ;
                        if (k < 0)
                        {
                            data [i][j+4] = " ?" ;
                        }
                        else if (k == 0)
                        {
                            data [i][j+4] = " no" ;
                        }
                        else
                        {
                            data [i][j+4] = " yes" ;
                        }
                    }
                    else
                    {
                        data [i][j+4] = newstats [i][j] ;
                    }
                }
            }
            fireTableDataChanged ( ) ;
        }
    }

    //--------------------------------------------------------------------------
    // get a value from the matrix table
    //--------------------------------------------------------------------------

    private Object get_table_value (int id, int j)
    {
        // id is in the range 1 to Stats.length.  The model index is id-1.
        // Convert this to the row index of the view and then get the data.
        // j is in the range 0 to 16, and is the same in the view and the
        // model, since column rearranging is never done.

        int i = matrix_Table.convertRowIndexToView (id-1) ;
        return (matrix_Table.getValueAt (i, j)) ;
    }

    //--------------------------------------------------------------------------
    // set a value in the matrix table
    //--------------------------------------------------------------------------

    private void set_table_value (Object value, int id, int j)
    {
        // just like get_table_value, setting the data instead of getting it

        int i = matrix_Table.convertRowIndexToView (id-1) ;
        matrix_Table.setValueAt (value, i, j) ;
    }

    //--------------------------------------------------------------------------
    // get ids of highlighted matrices
    //--------------------------------------------------------------------------

    private int [ ] get_highlighted_ids ( )
    {
        // return a list of highlighted matrix id's

        // get the highlighted row indices in the current view
        int [ ] highlighted = matrix_Table.getSelectedRows ( ) ;
        // convert the row view indices to matrix id's
        for (int k = 0 ; k < highlighted.length ; k++)
        {
            int i = highlighted [k] ;
            int id = 1 + matrix_Table.convertRowIndexToModel (i) ;
            highlighted [k] = id ;
        }
        return (highlighted) ;
    }

    //--------------------------------------------------------------------------
    // get ids of matrices selected for download
    //--------------------------------------------------------------------------

    private int [ ] get_download_ids ( )
    {
        // get the list of ids to download, in view order
        nselected = 0 ;
        for (int i = 0 ; i < Stats.length ; i++)
        {
            if ((Boolean) matrix_Table.getValueAt (i, 0))
            {
                nselected++ ;
            }
        }
        int [ ] downloads = new int [nselected] ;
        int k = 0 ;
        for (int i = 0 ; i < Stats.length ; i++)
        {
            if ((Boolean) matrix_Table.getValueAt (i, 0))
            {
                int id = 1 + matrix_Table.convertRowIndexToModel (i) ;
                downloads [k++] = id ;
            }
        }
        return (downloads) ;
    }

    //--------------------------------------------------------------------------
    // set "Matrices selected:" label and download tool tip
    //--------------------------------------------------------------------------

    private void set_selected_label (boolean enabled)
    {
        if (gui_ready)
        {
            nselected_Label.setText
                ("   Matrices selected: " + nselected + "   ") ;
            download_Button.setEnabled (enabled) ;
            if (enabled)
            {
                if (nselected == 0)
                {
                    download_Button.setToolTipText
                        ("No matrices have been selected for download") ;
                }
                else if (nselected == 1)
                {
                    download_Button.setToolTipText
                        ("Click to download the single selected matrix") ;
                }
                else
                {
                    download_Button.setToolTipText
                        ("Click to download the " + nselected +
                        " selected matrices") ;
                }
            }
            else
            {
                download_Button.setToolTipText ("Download in progress.") ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // show matrix icon
    //--------------------------------------------------------------------------

    private void show_highlighted_icon ( )
    {
        // show icon of last entry in the highlighted list
        int [ ] highlighted = get_highlighted_ids ( ) ;
        int n = highlighted.length ;
        if (n > 0)
        {
            int id = highlighted [n-1] ;
            String Group = (String) Stats [id-1][1] ;
            String Name  = (String) Stats [id-1][2] ;
            update_icon (Group + "/" + Name) ;
        }
    }

    //--------------------------------------------------------------------------
    // matrix table popup listener
    //--------------------------------------------------------------------------

    private class matrix_Table_PopupListener extends MouseAdapter
    {
        JPopupMenu pop ;

        matrix_Table_PopupListener (JPopupMenu popupMenu)
        {
            pop = popupMenu ;
        }

        public void mousePressed (MouseEvent e)
        {
            maybeShowPopup (e) ;
        }

        public void mouseReleased (MouseEvent e)
        {
            maybeShowPopup (e) ;
        }

        private void maybeShowPopup (MouseEvent e)
        {
            if (e.isPopupTrigger ( ))
            {
                pop.show (e.getComponent ( ), e.getX ( ), e.getY ( )) ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // matrix table row listener
    //--------------------------------------------------------------------------

    private class matrix_Table_RowListener implements ListSelectionListener
    {
        public void valueChanged (ListSelectionEvent event)
        {
            if (event.getValueIsAdjusting ( ))
            {
                return ;
            }
            show_highlighted_icon ( ) ;
        }
    }

    //--------------------------------------------------------------------------
    // FindKinds:  determine the set of all Problem kinds
    //--------------------------------------------------------------------------

    private String [ ] FindKinds ( )
    {
        Set<String> KindSet = new TreeSet<String> ( ) ;
        KindSet.add (all_kinds) ;
        for (int id = 1 ; id <= Stats.length ; id++)
        {
            KindSet.add (SimplifyKind ((String) Stats [id-1][12])) ;
        }
        return ((String [ ]) KindSet.toArray (new String [0])) ;
    }

    //--------------------------------------------------------------------------
    // FindGroups:  determine the set of all groups
    //--------------------------------------------------------------------------

    private String [ ] FindGroups ( )
    {
        Set<String> GroupSet = new TreeSet<String> ( ) ;
        GroupSet.add (all_groups) ;
        for (int id = 1 ; id <= Stats.length ; id++)
        {
            GroupSet.add ((String) Stats [id-1][1]) ;
        }
        return ((String [ ]) GroupSet.toArray (new String [0])) ;
    }

    //--------------------------------------------------------------------------
    // SimplifyKind: remove extranneous terms from a string
    //--------------------------------------------------------------------------

    private String SimplifyKind (String kind)
    {
        // remove terms from a matrix-kind string
        String result = null ;
        String [ ] token = kind.split (" ") ;
        for (int i = 0 ; i < token.length ; i++)
        {
            if (! (token [i].equals ("subsequent")
                || token [i].equals ("sequence")
                || token [i].equals ("problem")
                || token [i].equals ("duplicate")))
            {
                if (result == null)
                {
                    result = token [i] ;
                }
                else
                {
                    result = result + " " + token [i] ;
                }
            }
        }
        return (result) ;
    }

    //--------------------------------------------------------------------------
    // CheckDir:  return a directory, creating it if it doesn't exist
    //--------------------------------------------------------------------------

    private File CheckDir (String directory_name)
    {
        File dir = new File (fix_name (directory_name)) ;
        if (!dir.isDirectory ( ))
        {
            dir.mkdirs ( ) ;
        }
        return (dir) ;
    }

    //--------------------------------------------------------------------------
    // CheckExistence: determine which files exist in the local file system
    //--------------------------------------------------------------------------

    private void CheckExistence ( )
    {
        // check the existence all matrices in all 3 formats
        fire_updates (false) ;
        for (int id = 1 ; id <= Stats.length ; id++)
        {
            CheckExistence (id) ;
        }
        fire_updates (true) ;
    }

    private boolean [ ] CheckExistence (int id)
    {
        // check the existence of a single file (in all 3 formats)
        boolean [ ] exists = new boolean [4] ;
        boolean [ ] etable = new boolean [3] ;

        String Group = (String) Stats [id-1][1] ;
        String Name  = (String) Stats [id-1][2] ;

        for (int j = 0 ; j < 3 ; j++)
        {
            etable [j] = (((String) get_table_value (id, j+1)).charAt (0) == 'x') ;
        }

        for (int j = 0 ; j < 4 ; j++)
        {
            exists [j] = false ;
        }

        // check for mat/HB/west0067.mat
        File G = new File (mat, Group) ;
        if (G.isDirectory ( ) && (new File (G, Name + ".mat")).exists ( ))
        {
            exists [0] = true ;
        }

        // check for MM/HB/west0067.tar.gz
        G = new File (MM, Group) ;
        if (G.isDirectory ( ) && (new File (G, Name + ".tar.gz")).exists ( ))
        {
            exists [1] = true ;
        }

        // check for MM/HB/west0067.tar.gz
        G = new File (RB, Group) ;
        if (G.isDirectory ( ) && (new File (G, Name + ".tar.gz")).exists ( ))
        {
            exists [2] = true ;
        }

        // check for files/HB/west0067.png
        G = new File (iconDir, Group) ;
        if (G.isDirectory ( ) && (new File (G, Name + ".png")).exists ( ))
        {
            exists [3] = true ;
        }

        // update the matrix table (mat, MM, and RB columns)
        for (int j = 0 ; j < 3 ; j++)
        {
            if (etable [j] != exists [j])
            {
                // only update the table if the existence status has changed
                set_table_value (exists [j] ? "x" : "-", id, j+1) ;
            }
        }
        return (exists) ;
    }

    //-------------------------------------------------------------------------
    // get long from JTextField
    //-------------------------------------------------------------------------

    private long getLong (JTextField tfield, long Default)
    {
        String s = tfield.getText ( ) ;
        long result = Default ;
        if (s.length ( ) > 0)
        {
            try
            {
                result = Long.parseLong (s) ;
            }
            catch (Exception e)
            {
            }
        }
        return (result) ;
    }

    //-------------------------------------------------------------------------
    // get double from JTextField
    //-------------------------------------------------------------------------

    private double getDouble (JTextField tfield, double Default)
    {
        String s = tfield.getText ( ) ;
        double result = Default ;
        if (s.length ( ) > 0)
        {
            try
            {
                result = Double.parseDouble (s) ;
            }
            catch (Exception e)
            {
            }
        }
        return (result) ;
    }

    //-------------------------------------------------------------------------
    // change to a wait cursor
    //-------------------------------------------------------------------------

    private void please_wait ( )
    {
        this.setCursor (Cursor.getPredefinedCursor (Cursor.WAIT_CURSOR)) ;
    }

    //-------------------------------------------------------------------------
    // change to a normal cursor
    //-------------------------------------------------------------------------

    private void the_long_wait_is_over ( )
    {
        this.setCursor (Cursor.getDefaultCursor ( )) ;
    }

    //-------------------------------------------------------------------------
    // make or clear a selection
    //-------------------------------------------------------------------------

    private void make_selection (boolean action)
    {
        // set selections according to controls
        please_wait ( ) ;
        fire_updates (false) ;

        long minrow = getLong (minrow_Field, 0) ;
        long maxrow = getLong (maxrow_Field, INF) ;

        long mincol = getLong (mincol_Field, 0) ;
        long maxcol = getLong (maxcol_Field, INF) ;

        long minnz = getLong (minnz_Field, 0) ;
        long maxnz = getLong (maxnz_Field, INF) ;

        double minpsym = getDouble (minpsym_Field, 0) ;
        double maxpsym = getDouble (maxpsym_Field, 1.0) ;

        double minnsym = getDouble (minnsym_Field, 0) ;
        double maxnsym = getDouble (maxnsym_Field, 1.0) ;

        boolean shape_square = shape_square_Button.isSelected ( ) ;
        boolean shape_rect   = shape_rect_Button.isSelected ( ) ;
        boolean shape_either = shape_either_Button.isSelected ( ) ;

        boolean posdef_yes    = posdef_yes_Button.isSelected ( ) ;
        boolean posdef_no     = posdef_no_Button.isSelected ( ) ;
        boolean posdef_either = posdef_either_Button.isSelected ( ) ;

        boolean nd_yes    = nd_yes_Button.isSelected ( ) ;
        boolean nd_no     = nd_no_Button.isSelected ( ) ;
        boolean nd_either = nd_either_Button.isSelected ( ) ;

        boolean real_yes    = real_yes_Button.isSelected ( ) ;
        boolean real_no     = real_no_Button.isSelected ( ) ;
        boolean real_either = real_either_Button.isSelected ( ) ;

        // create HashSet for the selected groups
        Set<String> Gset = null ;
        Object [ ] Gsel = Group_List.getSelectedValuesList ( ).toArray ( ) ;
        int ngroups = Gsel.length ;
        if (ngroups > 0)
        {
            for (int i = 0 ; i < ngroups ; i++)
            {
                if (((String) Gsel [i]).equals (all_groups)) ngroups = 0 ;
            }
            Gset = new HashSet<String> ( ) ;
            for (int i = 0 ; i < ngroups ; i++)
            {
                Gset.add ((String) Gsel [i]) ;
                if (debug) System.out.println ("Group: " + (String) Gsel [i]) ;
            }
        }

        // create HashSet for the selected kinds
        Set<String> Kset = null ;
        Object [ ] Ksel = Kind_List.getSelectedValuesList ( ).toArray ( ) ;
        int nkinds = Ksel.length ;
        if (nkinds > 0)
        {
            for (int i = 0 ; i < nkinds ; i++)
            {
                if (((String) Ksel [i]).equals (all_kinds)) nkinds = 0 ;
            }
            Kset = new HashSet<String> ( ) ;
            for (int i = 0 ; i < nkinds ; i++)
            {
                Kset.add ((String) Ksel [i]) ;
                if (debug) System.out.println ("Kind: " + ((String) Ksel [i])) ;
            }
        }

        for (int id = 1 ; id <= Stats.length ; id++)
        {

            // look at the matrix properties to see if it fits the selection
            long nrows = (Long) Stats [id-1][3] ;
            long ncols = (Long) Stats [id-1][4] ;
            long nnz   = (Long) Stats [id-1][5] ;

            int isReal   = (Integer) Stats [id-1][6] ;
            int isBinary = (Integer) Stats [id-1][7] ;
            int isND     = (Integer) Stats [id-1][8] ;
            int posdef   = (Integer) Stats [id-1][9] ;

            double psym = (Double) Stats [id-1][10] ;
            double nsym = (Double) Stats [id-1][11] ;

            boolean choose_group = true ;
            if (ngroups > 0)
            {
                String group = (String) Stats [id-1][1] ;
                choose_group = Gset.contains (group) ;
            }

            boolean choose_kind = true ;
            if (nkinds > 0)
            {
                String kind = SimplifyKind ((String) Stats [id-1][12]) ;
                choose_kind = Kset.contains (kind) ;
            }

            if ((minrow <= nrows && nrows <= maxrow) &&
                (mincol <= ncols && ncols <= maxcol) &&
                (minnz <= nnz && nnz <= maxnz) &&
                (minpsym <= psym && psym <= maxpsym) &&
                (minnsym <= nsym && nsym <= maxnsym) &&
                (posdef_either ||
                    (posdef_yes && posdef == 1) ||
                    (posdef_no && posdef == 0)) &&
                (nd_either ||
                    (nd_yes && isND == 1) ||
                    (nd_no && isND == 0)) &&
                (real_either ||
                    (real_yes && isReal == 1) ||
                    (real_no && isReal == 0)) &&
                (shape_either ||
                    (shape_square && nrows == ncols) ||
                    (shape_rect && nrows != ncols)) &&
                choose_group && choose_kind)
            {
                // change the selection box for this matrix id
                set_table_value (action, id, 0) ;
            }
        }
        fire_updates (true) ;
        progress1_Bar.setValue (0) ;
        progress2_Bar.setValue (0) ;
        the_long_wait_is_over ( ) ;
    }

    //-------------------------------------------------------------------------
    // reset button
    //-------------------------------------------------------------------------

    private void reset_Button_action (ActionEvent e)
    {
        // reset the selection criteria to the defaults
        minrow_Field.setText ("") ;
        maxrow_Field.setText ("") ;

        mincol_Field.setText ("") ;
        maxcol_Field.setText ("") ;

        minnz_Field.setText ("") ;
        maxnz_Field.setText ("") ;

        minpsym_Field.setText ("0.0") ;
        maxpsym_Field.setText ("1.0") ;

        minnsym_Field.setText ("0.0") ;
        maxnsym_Field.setText ("1.0") ;

        shape_either_Button.setSelected (true) ;
        posdef_either_Button.setSelected (true) ;
        nd_either_Button.setSelected (true) ;
        real_either_Button.setSelected (true) ;

        Group_List.clearSelection ( ) ;
        Kind_List.clearSelection ( ) ;

        progress1_Bar.setValue (0) ;
        progress2_Bar.setValue (0) ;
    }

    //-------------------------------------------------------------------------
    // clear button
    //-------------------------------------------------------------------------

    private void clear_Button_action (ActionEvent e)
    {
        // set selections according to controls
        please_wait ( ) ;
        fire_updates (false) ;

        for (int id = 1 ; id <= Stats.length ; id++)
        {
            // clear the selection box for this matrix id
            set_table_value (false, id, 0) ;
        }
        fire_updates (true) ;
        progress1_Bar.setValue (0) ;
        progress2_Bar.setValue (0) ;
        the_long_wait_is_over ( ) ;
    }

    //-------------------------------------------------------------------------
    // select popup menu item
    //-------------------------------------------------------------------------

    private void popup_action (ActionEvent e, boolean action)
    {
        // select or deselect the highlight matrices
        please_wait ( ) ;
        int [ ] highlighted = get_highlighted_ids ( ) ;
        int n = highlighted.length ;
        for (int k = 0 ; k < n ; k++)
        {
            set_table_value (action, highlighted [k], 0) ;
        }
        the_long_wait_is_over ( ) ;
    }

    //-------------------------------------------------------------------------
    // export popup menu item
    //-------------------------------------------------------------------------

    private void export_list_action (ActionEvent e, boolean csv)
    {
        // export the list in the order of the current view
        if (chooser.showSaveDialog (ssgui.this) == JFileChooser.APPROVE_OPTION)
        {
            please_wait ( ) ;
            print_out = null ;
            try
            {
                print_out = new PrintWriter (chooser.getSelectedFile ( )) ;
                int [ ] ids = get_download_ids ( ) ;
                int n = ids.length ;

                // print the header
                if (csv)
                {
                    print_out.println ("mat, MM, RB, id, Group, Name, rows, " +
                        "cols, nonzeros, real, binary, 2D/3D, posdef, psym, " +
                        "nsym, kind") ;
                }
                else
                {
                    print_out.println ("%% Matrices selected from ssgui:") ;
                    print_out.println ("% Example usage:") ;
                    print_out.println ("% for k = 1:length(ids)") ;
                    print_out.println ("%    Problem = ssget (ids (k))") ;
                    print_out.println ("% end") ;
                    print_out.println ("ids = [")  ;
                }

                for (int k = 0 ; k < n ; k++)
                {
                    // get the matrix id and stats
                    int id = ids [k] ;
                    boolean [ ] exists = CheckExistence (id) ;
                    String Group       = (String)  Stats [id-1][1] ;
                    String Name        = (String)  Stats [id-1][2] ;
                    long nrows         = (Long)    Stats [id-1][3] ;
                    long ncols         = (Long)    Stats [id-1][4] ;
                    long nnz           = (Long)    Stats [id-1][5] ;
                    int isReal         = (Integer) Stats [id-1][6] ;
                    int isBinary       = (Integer) Stats [id-1][7] ;
                    int isND           = (Integer) Stats [id-1][8] ;
                    int posdef         = (Integer) Stats [id-1][9] ;
                    double psym        = (Double)  Stats [id-1][10] ;
                    double nsym        = (Double)  Stats [id-1][11] ;
                    String kind        = (String)  Stats [id-1][12] ;

                    if (csv)
                    {
                        // print the matrix stats in a single CSV line of text
                        print_out.println (
                            exists [0] + ", " + exists [1] + ", " +
                            exists [2] + ", " + id + ", " + Group + ", " +
                            Name + ", " + nrows + ", " + ncols + ", " +
                            nnz + ", " + isReal + ", " + isBinary + ", " +
                            isND + ", " + posdef + ", " + psym + ", " +
                            nsym + ", " + kind) ;
                    }
                    else
                    {
                        // print the matrix id, with a comment for the name
                        print_out.println (id + " % " + Group + "/" + Name) ;
                    }
                }
                if (!csv)
                {
                    print_out.println ("] ;")  ;
                }
            }
            catch (Exception err)
            {
                // display warning dialog
                JOptionPane.showMessageDialog (this, "Export failed.",
                    "Warning", JOptionPane.WARNING_MESSAGE) ;
            }
            finally
            {
                close_printer_stream (print_out) ;
            }
            the_long_wait_is_over ( ) ;
        }
    }

    //-------------------------------------------------------------------------
    // help button
    //-------------------------------------------------------------------------

    private void help_Button_action (ActionEvent e)
    {
        // create the Help window
        please_wait ( ) ;
        JFrame help_Frame = new JFrame ("Help: SuiteSparse Matrix Collection") ;

        // open the HTML help file and put it in an editor pane
        JEditorPane editorPane = new JEditorPane ( ) ;
        editorPane.setEditable (false) ;
        URL helpURL = ssgui.class.getResource ("sshelp.html") ;
        if (helpURL != null)
        {
            try
            {
                editorPane.setPage (helpURL) ;
            }
            catch (IOException error)
            {
                // display warning dialog
                JOptionPane.showMessageDialog (this,
                    "Sorry, Help document sshelp.html not found.",
                    "Warning", JOptionPane.WARNING_MESSAGE) ;
            }
        }

        // Put the editor pane in a scroll pane.
        JScrollPane editorScrollPane = new JScrollPane (editorPane) ;

        // Add the scroll pane to the Help window
        help_Frame.getContentPane ( ).add (editorScrollPane) ;
        help_Frame.setSize (800,600) ;
        help_Frame.setVisible (true) ;

        the_long_wait_is_over ( ) ;
    }

    //-------------------------------------------------------------------------
    // get the icon filename
    //-------------------------------------------------------------------------

    private String icon_file (String fullname)
    {
        return ("files/" + fullname + ".png") ;
    }

    //-------------------------------------------------------------------------
    // update the icon
    //-------------------------------------------------------------------------

    private void update_icon (String fullname)
    {
        // fullname is of the form Group/Name (HB/west0479, for example)
        icon_Label.setText (fullname) ;
        ImageIcon icon = new ImageIcon (fix_name (icon_file (fullname))) ;
        if (icon.getIconWidth ( ) < 0)
        {
            // icon image failed to load; get the image from the web
            icon = new ImageIcon (get_url (sssite +"/"+ icon_file (fullname))) ;
        }
        icon_Label.setIcon (icon) ;
    }

    //--------------------------------------------------------------------------
    // cancel button
    //--------------------------------------------------------------------------

    private void cancel_Button_action (ActionEvent e)
    {
        if (downloading && !cancel)
        {
            cancel = true ;
            cancel_Button.setEnabled (false) ;
            cancel_Button.setToolTipText ("canceling...") ;
        }
    }

    //-------------------------------------------------------------------------
    // get all icons
    //-------------------------------------------------------------------------

    private void get_all_icons ( )
    {
        // get all icons
        start_download_thread (0) ;
    }

    //-------------------------------------------------------------------------
    // download button
    //-------------------------------------------------------------------------

    private void download_Button_action (ActionEvent e)
    {
        // get the selected matrices
        start_download_thread (2) ;
    }

    //-------------------------------------------------------------------------
    // start the downloader thread
    //-------------------------------------------------------------------------

    private void start_download_thread (int what)
    {
        if (!downloading)
        {
            // only allow one active download at a time
            downloading = true ;
            cancel = false ;

            if (gui_ready)
            {
                cancel_Button.setEnabled (true) ;
                cancel_Button.setToolTipText
                    ("Click to cancel the current download.") ;
            }

            if (what == 0)
            {
                // get all the icons
                get_icons = true ;
                download_ids = null ;
            }
            else
            {
                // download one or more matrices
                get_icons = false ;
                download_ids = get_download_ids ( ) ;
            }
            set_selected_label (false) ;
            // start the downloader thread
            ssdownload tt = new ssdownload ( ) ;
        }
    }

    //--------------------------------------------------------------------------
    // downloader thread
    //--------------------------------------------------------------------------

    private class ssdownload implements Runnable
    {

        // constructor starts the downloader thread
        public ssdownload ( )
        {
            Thread thread = new Thread (this) ;
            thread.start ( ) ;
        }

        // thread.start calls the run method
        public void run ( )
        {

            if (get_icons)
            {
                // get all missing icons
                progress1_Bar.setValue (1) ;
                progress1_Bar.setMaximum (Stats.length) ;
                icon_Label.setBorder (BorderFactory.createTitledBorder
                    ("checking for new matrix icons")) ;
                for (int id = 1 ; !cancel && id <= Stats.length ; id++)
                {
                    boolean [ ] exists = CheckExistence (id) ;
                    if (!exists [3])
                    {
                        icon_Label.setBorder (BorderFactory.createTitledBorder
                            ("downloading new matrix icons")) ;
                        String Group = (String) Stats [id-1][1] ;
                        String Name  = (String) Stats [id-1][2] ;
                        String fullname = Group + "/" + Name ;
                        CheckDir ("files/" + Group) ;
                        download_file (icon_file (fullname)) ;
                        update_icon (fullname) ;
                    }
                    progress1_Bar.setValue (id+2) ;
                }
                progress1_Bar.setValue (Stats.length) ;
                icon_Label.setBorder (BorderFactory.createTitledBorder
                    ("matrix icon")) ;
            }

            if (download_ids != null && download_ids.length > 0)
            {
                // download all selected matrices in the requested formats

                // determine which formats to download
                int barmax = download_ids.length + 2 ;

                boolean format_mat = format_mat_Button.isSelected ( ) ;
                boolean format_mm  = format_mm_Button.isSelected ( ) ;
                boolean format_rb  = format_rb_Button.isSelected ( ) ;

                // start the overall progress bar
                progress1_Bar.setValue (1) ;
                progress1_Bar.setMaximum (barmax) ;

                // download all the files
                for (int k = 0 ; !cancel && k < download_ids.length ; k++)
                {
                    int id = download_ids [k] ;

                    // get matrxx
                    String Group = (String) Stats [id-1][1] ;
                    String Name  = (String) Stats [id-1][2] ;
                    String fullname = Group + "/" + Name ;

                    // recheck to see if the matrix exists in the 4 formats
                    boolean [ ] exists = CheckExistence (id) ;

                    if (!exists [3])
                    {
                        // always download the matrix icon if it doesn't exist
                        CheckDir ("files/" + Group) ;
                        download_file (icon_file (fullname)) ;
                        update_icon (fullname) ;
                    }

                    if (!exists [0] && format_mat)
                    {
                        // download the matrix in MATLAB format
                        update_icon (fullname) ;
                        CheckDir ("mat/" + Group) ;
                        download_file ("mat/" + fullname + ".mat") ;
                    }

                    if (!exists [1] && format_mm)
                    {
                        // download the matrix in Matrix Market format
                        update_icon (fullname) ;
                        CheckDir ("MM/" + Group) ;
                        download_file ("MM/" + fullname + ".tar.gz") ;
                    }

                    if (!exists [2] && format_rb)
                    {
                        // download the matrix in Rutherford/Boeing format
                        update_icon (fullname) ;
                        CheckDir ("RB/" + Group) ;
                        download_file ("RB/" + fullname + ".tar.gz") ;
                    }

                    progress1_Bar.setValue (k+2) ;
                }

                // update the mat/MM/RB check boxes
                for (int k = 0 ; k < download_ids.length ; k++)
                {
                    int id = download_ids [k] ;
                    CheckExistence (id) ;
                }

                // finish the overall progress bar
                progress1_Bar.setValue (barmax) ;
            }

            cancel_Button.setEnabled (false) ;
            cancel_Button.setToolTipText ("No downloads in progress.") ;

            set_selected_label (true) ;
            cancel = false ;
            downloading = false ;
        }
    }

    //--------------------------------------------------------------------------
    // get a URL
    //--------------------------------------------------------------------------

    private URL get_url (String urlstring)
    {
        try
        {
            return (new URL (urlstring)) ;
        }
        catch (MalformedURLException e)
        {
            // display warning dialog
            JOptionPane.showMessageDialog (this, "Invalid URL: "
                + urlstring, "Warning", JOptionPane.WARNING_MESSAGE) ;
            return (null) ;
        }
    }

    //--------------------------------------------------------------------------
    // download a file
    //--------------------------------------------------------------------------

    private void download_file (String filename)
    {
        boolean ok = true ;
        if (cancel) return ;
        String urlstring = sssite + "/" + filename ;
        if (debug) System.out.println ("downloading: " + urlstring) ;

        // create the URL
        URL url = get_url (urlstring) ;
        if (url == null) return ;

        // download the file
        url_in = null ;
        ftemp_out = null ;
        ftemp_name = filename + "_IN_PROGREss" ;
        int barmax = 1 ;

        try
        {
            // determine the file size (fails for files > 2GB)
            int len = url.openConnection ( ).getContentLength ( ) ;

            // start the progress bar
            if (gui_ready)
            {
                if (len < 0)
                {
                    progress2_Bar.setIndeterminate (true) ;
                    progress_size_Label.setText ("") ;
                }
                else
                {
                    progress2_Bar.setValue (0) ;
                    // display the filesize to the right of the progress bar
                    if (len < M)
                    {
                        barmax = 1 + len / K ;
                        progress_size_Label.setText (((len+K/2) / K) + " KB") ;
                    }
                    else
                    {
                        barmax = 1 + len / M ;
                        progress_size_Label.setText (((len+M/2) / M) + " MB") ;
                    }
                }
                progress2_Bar.setMaximum (barmax) ;
            }

            // open the source and destination files
            url_in = new BufferedInputStream (url.openStream ( )) ;
            ftemp_out = new BufferedOutputStream (new FileOutputStream
                (fix_name (ftemp_name)), buffersize) ;

            // transfer the data
            byte buffer [ ] = new byte [buffersize] ;
            long bytes_read = 0 ;
            int count = 0 ;
            while (!cancel && (count = url_in.read (buffer, 0, buffersize)) >= 0)
            {
                if (ftemp_out != null) ftemp_out.write (buffer, 0, count) ;
                bytes_read += count ;
                if (gui_ready && len > 0)
                {
                    if (len < M)
                    {
                        progress2_Bar.setValue ((int) (bytes_read / K)) ;
                    }
                    else
                    {
                        progress2_Bar.setValue ((int) (bytes_read / M)) ;
                    }
                }
            }
        }
        catch (Exception e)
        {
            // display warning dialog
            JOptionPane.showMessageDialog (this, "Download failed: "
                + urlstring, "Warning", JOptionPane.WARNING_MESSAGE) ;
            ok = false ;
        }

        if (gui_ready)
        {
            progress2_Bar.setIndeterminate (false) ;
            progress2_Bar.setValue (barmax) ;
            progress_size_Label.setText ("") ;
        }

        // wrap-up
        if (ok && !cancel)
        {
            // success:  rename the temp file to the permanent filename
            cleanup (false) ;
            File fsrc = new File (fix_name (ftemp_name)) ;
            File fdst = new File (fix_name (filename)) ;
            fsrc.renameTo (fdst) ;
        }
        else
        {
            // cancelled, or something failed:  delete the files if they exist
            cleanup (true) ;
        }
    }

    //--------------------------------------------------------------------------
    // download the latest matrix stats
    //--------------------------------------------------------------------------

    private Object [ ][ ] download_matrix_stats ( )
    {
        download_file (ssindex) ;       // download files/ss_index.mat
        download_file (ssstats) ;       // download files/ssstats.csv
        return (load_ssstats ( )) ;     // load the ssstats.csv file
    }

    //--------------------------------------------------------------------------
    // prepend the ss rchive directory and replace '/' with the file separator
    //--------------------------------------------------------------------------

    private static String fix_name (String s)
    {
        // file separator is '/' on Unix/Solaris/Linux/Mac, and '\' on Windows
        String r = ssarchive ;
        if (s != null)
        {
            r = r + s ;
        }
        return (r.replace ('/', File.separatorChar)) ;
    }

    //--------------------------------------------------------------------------
    // close an output stream
    //--------------------------------------------------------------------------

    private static void close_output (OutputStream out)
    {
        try
        {
            if (out != null) out.close ( ) ;
        }
        catch (IOException e)
        {
        }
    }

    //--------------------------------------------------------------------------
    // close an input stream
    //--------------------------------------------------------------------------

    private static void close_reader (Reader in)
    {
        try
        {
            if (in != null) in.close ( ) ;
        }
        catch (IOException e)
        {
        }
    }

    //--------------------------------------------------------------------------
    // close a printer stream
    //--------------------------------------------------------------------------

    private static void close_printer_stream (PrintWriter out)
    {
        if (out != null) out.close ( ) ;
    }

    //--------------------------------------------------------------------------
    // delete a file
    //--------------------------------------------------------------------------

    private static void delete_file (String filename)
    {
        if (filename != null)
        {
            File ff = new File (fix_name (filename)) ;
            if (ff.exists ( )) ff.delete ( ) ;
        }
    }

    //--------------------------------------------------------------------------
    // cleanup
    //--------------------------------------------------------------------------

    private static void cleanup (boolean delete)
    {
        // close input streams, if any
        try
        {
            if (url_in != null) url_in.close ( ) ;
        }
        catch (IOException e)
        {
        }
        url_in = null ;

        // close temporary file
        close_output (ftemp_out) ;
        ftemp_out = null ; 

        if (delete)
        {
            // delete temporary file
            delete_file (ftemp_name) ;
            ftemp_name = null ;
        }

        // close the printer stream, if any
        close_printer_stream (print_out) ;
        
        // close the reader stream, if any
        close_reader (in_reader) ;
    }

    //--------------------------------------------------------------------------
    // main method
    //--------------------------------------------------------------------------

    public static void main (String args [ ])
    {
        // register a shutdown hook
        Runtime.getRuntime ( ).addShutdownHook
        (
            new Thread ( )
            {
                public void run ( )
                {
                    // delete any temporary files when the ssgui shuts down,
                    // and close any files
                    cleanup (true) ;
                }
            }
        ) ;

        // start the GUI in its own thread
        EventQueue.invokeLater
        (
            new Runnable ( )
            {
                public void run ( )
                {
                    new ssgui ( ).setVisible (true) ;
                }
            }
        ) ;
    }
}
