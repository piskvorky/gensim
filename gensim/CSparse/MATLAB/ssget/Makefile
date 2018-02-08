ssgui.jar:  ssgui.java sshelp.html
	javac ssgui.java
	jar cfe ssgui.jar ssgui *.class sshelp.html
	- $(RM) *.class

run: ssgui.jar
	java -jar ssgui.jar

clean:
	- $(RM) *.class

purge:
	- $(RM) *.class

check:
	javac -Xlint:deprecation -Xlint:unchecked ssgui.java
