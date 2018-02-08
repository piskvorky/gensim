
# deprecated cimport for backwards compatibility:
from libc.string cimport const_char


cdef extern from "<string>" namespace "std" nogil:

    size_t npos = -1

    cdef cppclass string:
        string() except +
        string(char *) except +
        string(char *, size_t) except +
        string(string&) except +
        # as a string formed by a repetition of character c, n times.
        string(size_t, char) except +

        cppclass iterator:
            iterator()
            char& operator*()
            iterator(iterator &)
            iterator operator++()
            iterator operator--()
            bint operator==(iterator)
            bint operator!=(iterator)
        cppclass reverse_iterator:
            char& operator*()
            iterator operator++()
            iterator operator--()
            iterator operator+(size_t)
            iterator operator-(size_t)
            bint operator==(reverse_iterator)
            bint operator!=(reverse_iterator)
            bint operator<(reverse_iterator)
            bint operator>(reverse_iterator)
            bint operator<=(reverse_iterator)
            bint operator>=(reverse_iterator)
        cppclass const_iterator(iterator):
            pass
        cppclass const_reverse_iterator(reverse_iterator):
            pass

        iterator begin()
        const_iterator const_begin "begin"()
        iterator end()
        const_iterator const_end "end"()
        reverse_iterator rbegin()
        const_reverse_iterator const_rbegin "rbegin"()
        reverse_iterator rend()
        const_reverse_iterator const_rend "rend"()

        const char* c_str()
        const char* data()
        size_t size()
        size_t max_size()
        size_t length()
        void resize(size_t)
        void resize(size_t, char c)
        size_t capacity()
        void reserve(size_t)
        void clear()
        bint empty()

        char& at(size_t)
        char& operator[](size_t)
        int compare(string&)

        string& append(string&)
        string& append(string&, size_t, size_t)
        string& append(char *)
        string& append(char *, size_t)
        string& append(size_t, char)

        void push_back(char c)

        string& assign (string&)
        string& assign (string&, size_t, size_t)
        string& assign (char *, size_t)
        string& assign (char *)
        string& assign (size_t n, char c)

        string& insert(size_t, string&)
        string& insert(size_t, string&, size_t, size_t)
        string& insert(size_t, char* s, size_t)


        string& insert(size_t, char* s)
        string& insert(size_t, size_t, char c)

        size_t copy(char *, size_t, size_t)

        size_t find(string&)
        size_t find(string&, size_t)
        size_t find(char*, size_t pos, size_t)
        size_t find(char*, size_t pos)
        size_t find(char, size_t pos)

        size_t rfind(string&, size_t)
        size_t rfind(char* s, size_t, size_t)
        size_t rfind(char*, size_t pos)
        size_t rfind(char c, size_t)
        size_t rfind(char c)

        size_t find_first_of(string&, size_t)
        size_t find_first_of(char* s, size_t, size_t)
        size_t find_first_of(char*, size_t pos)
        size_t find_first_of(char c, size_t)
        size_t find_first_of(char c)

        size_t find_first_not_of(string&, size_t)
        size_t find_first_not_of(char* s, size_t, size_t)
        size_t find_first_not_of(char*, size_t pos)
        size_t find_first_not_of(char c, size_t)
        size_t find_first_not_of(char c)

        size_t find_last_of(string&, size_t)
        size_t find_last_of(char* s, size_t, size_t)
        size_t find_last_of(char*, size_t pos)
        size_t find_last_of(char c, size_t)
        size_t find_last_of(char c)

        size_t find_last_not_of(string&, size_t)
        size_t find_last_not_of(char* s, size_t, size_t)
        size_t find_last_not_of(char*, size_t pos)

        string substr(size_t, size_t)
        string substr()
        string substr(size_t)

        size_t find_last_not_of(char c, size_t)
        size_t find_last_not_of(char c)

        #string& operator= (string&)
        #string& operator= (char*)
        #string& operator= (char)

        string operator+ (string& rhs)
        string operator+ (char* rhs)

        bint operator==(string&)
        bint operator==(char*)

        bint operator!= (string& rhs )
        bint operator!= (char* )

        bint operator< (string&)
        bint operator< (char*)

        bint operator> (string&)
        bint operator> (char*)

        bint operator<= (string&)
        bint operator<= (char*)

        bint operator>= (string&)
        bint operator>= (char*)
