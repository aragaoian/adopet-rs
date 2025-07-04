<div align="center">
  <img 
    src="https://github.com/user-attachments/assets/705f8332-ed9e-48d9-9126-56a733c19a29" 
    alt="Adopet Logo" 
    style="width: 50%; height: auto; margin-bottom: 10px;" 
  />
  <p style="max-width: 600px;">
    <b>Adopet Rs</b>: A Feature-Weighted Content-Based filtering developed to decrease adopted pets abandonment and aid to democratize the adoption process
  </p>
  <a href="adopet.top">
    adopet.top
  </a>
</div>

# About ADOPET

Adopet was developed during the Software Engineering II course as part of an extension project. aiming to positively impact the community by applying knowledge acquired at the university. </br>

After analyzing other pet adoption websites, we were engaged to create a pet adoption website to help adopters find their ideal companion and reduce common adoption issues, such as pet abandonment, by recommending pets based on the user preferences.
We achieve this by using machine learning techniques and information gathered through a questionnaire completed during the registration process. 

# Details

Behind the scenes, a **Feature-Weighted Content-Based Filtering** approach was used, calculating the similarities with a cossine function and thus measuring how similar the user's and the item's profiles are. </br>

User Profille vector were multiplied by a vector of weigths to apply an augmentor of importance to some features, such as expense range, animal preference and housing size. Moreover, lower bound features were also submited to a process of augmentation and reduction of their values, allowing the recommender to recommend a lower cost pet even if the user chose a higher expense range. 

# Built With
- [**NumPy**](https://numpy.org/doc/) - Fundamental package for numerical computing with support for arrays, matrices, and mathematical operations.
- [**Pandas**](https://pandas.pydata.org/docs/) - Powerful data structures for data manipulation and analysis, especially for tabular data (DataFrames).
- [**Scikit-Learn**](https://scikit-learn.org/) - Widely used library for machine learning: classification, regression, clustering, dimensionality reduction, and more.
- [**FastAPI**](https://fastapi.tiangolo.com/) - High-performance web framework for building APIs with automatic OpenAPI docs. Based on Pydantic & Starlette.
- [**SQLAlchemy**](https://docs.sqlalchemy.org/) - Powerful SQL toolkit and Object-Relational Mapping (ORM) library for Python.

# Contributions

This project and the others ones that are part of the Adopet enviroment, were built alongside with my university collegues:
- [**Lucas Losekann**](https://github.com/lucaslosekann)
- [**João Fraga**](https://github.com/joaopedrofraga)
- [**Maria Martim**](https://github.com/MariaMartim)

# Other
- [**Adopet Back-end**](https://github.com/lucaslosekann/adopet-backend)
- [**Adopet Front-end**](https://github.com/lucaslosekann/adopet-frontend)


