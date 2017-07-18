{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}

   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :toctree: generated/methods

   {% for item in methods if (item == "assertArraySplitEqual") or ((item.find("assert") != 0) and (item.find("fail") != 0)) %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
      :toctree: generated/attribs
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
