# dialog2.tcl --
#
# This demonstration script creates a dialog box with a global grab.

after idle {
    .dialog2.msg configure -wraplength 4i
}
after 100 {
    grab -global .dialog2
}
set i [tk_dialog .dialog2 "Dialog with global grab" {This dialog box uses a global grab, so it prevents you from interacting with anything on your display until you invoke one of the buttons below.  Global grabs are almost always a bad idea; don't use them unless you're truly desperate.} warning 0 OK Cancel {Show Code}]

switch $i {
    0 {puts "You pressed OK"}
    1 {puts "You pressed Cancel"}
    2 {showCode .dialog2}
}
