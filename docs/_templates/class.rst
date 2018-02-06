:mod:`{{module}}`.{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    :members:

    {% block methods %}
    .. automethod:: __init__
    {% endblock %}

.. raw:: html

    <div class="clearer"></div>
