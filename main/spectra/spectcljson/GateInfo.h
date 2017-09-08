
#ifndef GATEINFO_H
#define GATEINFO_H

#include <vector>
#include <string>
#include <memory>

namespace SpJs
{

  enum GateType { SliceGate, ContourGate, BandGate, FalseGate, TrueGate, C2BandGate,
                  GammaSliceGate, GammaBandGate, GammaContourGate, EqualMaskGate, AndMaskGate,
                  NotMaskGate, NotGate, OrGate, AndGate, UnrecognizedGateType};

  struct GateInfo 
  {

  protected:
    std::string m_name;
    GateType    m_type;

  public:
    GateInfo(const std::string& name, GateType type);
    GateInfo(const GateInfo& rhs);

    virtual ~GateInfo();

    virtual GateType getType() const {          // Reasonable default impls.
      return m_type;
    }
    virtual std::unique_ptr<GateInfo> clone() const {
      return std::move(std::unique_ptr<GateInfo>(new GateInfo(*this)));
    };

    void setName(const std::string& name) { m_name = name; }
    std::string getName() const { return m_name; }

    virtual bool operator==(const GateInfo& rhs) const;
    virtual bool operator!=(const GateInfo& rhs) const;

  protected:

    void setType(GateType type) { m_type = type; }
  }; // end of GateInfo class


  struct GateInfo2D : public GateInfo
  {
    public:
      GateInfo2D(const std::string& name, GateType type);
      GateInfo2D(const GateInfo2D& rhs);
      virtual ~GateInfo2D();
      bool operator==(const GateInfo2D& rhs) const;
      bool operator!=(const GateInfo2D& rhs) const;

      virtual GateType getType() const = 0;
      virtual std::unique_ptr<GateInfo> clone() const = 0;

      virtual std::vector<std::pair<double, double> > getPoints() const = 0;
      virtual std::vector<std::pair<double, double> >& getPoints() = 0;
      virtual std::pair<double, double> getPoint(size_t index) const = 0;
      virtual void setPoints(const std::vector<std::pair<double, double> >& points) = 0;
      virtual void setPoint(size_t index, double x, double y) = 0;

      virtual std::string getParameter0() const = 0;
      virtual std::string getParameter1() const = 0;
  };

  // Forward declarations of classes

  class Slice;
  class Band;
  class Contour;

  /**
   * Cuts are gates with a low and a high limit.
   * They are the base for slices and gamma slices...both of which
   * have a low and a high limit value.
   */
  struct Cut : public GateInfo
  {
  private:
    double m_low;
    double m_high;
  
  public:
    Cut(GateType type);
    Cut(const std::string& name, GateType type, double low, double high);
    Cut(const Cut& rhs);
    virtual ~Cut();
    
    virtual std::unique_ptr<GateInfo> clone() const;
    
  public:
    void setLowerLimit(double value) { m_low = value; }
    double getLowerLimit() const { return m_low; }

    void setUpperLimit(double value) { m_high = value; }
    double getUpperLimit() const { return m_high; }
    
  public:
    virtual bool operator==(const Cut& rhs) const;
    virtual bool operator!=(const Cut& rhs) const;

  };

  /**! \brief Slices are 1d gates on a single parameter
   *
   */
  struct Slice : public Cut
  {
  private:
    std::string m_param;
  public:
    Slice();
    Slice(const std::string& name, const std::string& parameter, double low, double high);
    Slice(const Slice& rhs);
    virtual ~Slice();

    std::unique_ptr<GateInfo> clone() const;

    void setParameter(const std::string& paramName) { m_param = paramName; }
    std::string getParameter() const { return m_param; }

    virtual bool operator==(const Slice& rhs) const;
    virtual bool operator!=(const Slice& rhs) const;
  }; // end of class

  
  /**
   * @brief gamma slices are like slices but have multiple parameters:
   */
  struct GammaSlice : public Cut {
  private:
    std::vector<std::string> m_params;
  public:
    GammaSlice();
    GammaSlice(
        const std::string& name, const std::vector<std::string>& parameters,
        double low, double high
    );
    GammaSlice(const GammaSlice& rhs);
    virtual ~GammaSlice();
    
    std::unique_ptr<GateInfo> clone() const;
    
    void setParameter(const std::string& newParam, int idx);
    void appendParameter(const std::string& newParam);
    void setAllParameters(const std::vector<std::string>& parameters);
    
    const std::vector<std::string>& getParameters() const;
    std::string getParameter(int idx = 0) const;
    
    virtual bool operator==(const GammaSlice& rhs) const;
    virtual bool operator!=(const GammaSlice& rhs) const;
};                  // end class GammaSlice.
    

  /**! \brief Bands are 2d gates on two parameter (open)
   *
   */
  struct Band : public GateInfo2D
  {
  private:
    std::string m_param0;
    std::string m_param1;
    std::vector<std::pair<double, double> > m_points;

  public:
    Band();
    Band(const std::string& name,
            const std::string& param0,
            const std::string& param1,
            const std::vector<std::pair<double,double> >& points);
    Band(const GateInfo2D& contour);
    Band(const Band& rhs);
    virtual ~Band();

    GateType getType() const { return m_type; }
    std::unique_ptr<GateInfo> clone() const;

    void setParameter0(const std::string& paramName) { m_param0 = paramName; }
    std::string getParameter0() const { return m_param0; }

    void setParameter1(const std::string& paramName) { m_param1 = paramName; }
    std::string getParameter1() const { return m_param1; }

    std::vector<std::pair<double, double> > getPoints() const { return m_points; }
    std::vector<std::pair<double, double> >& getPoints() { return m_points; }
    std::pair<double, double> getPoint(size_t index) const
    { return m_points.at(index); }

    void setPoints(const std::vector<std::pair<double, double> >& points) {
        m_points = points;
    }
    void setPoint(size_t index, double x, double y);

    virtual bool operator==(const Band& rhs) const;
    virtual bool operator!=(const Band& rhs) const;
  }; // end of class


  /**! \brief Contours are 2d gates on a two parameters (closed areas)
   *
   */
  struct Contour : public GateInfo2D
  {
  private:
    std::string m_param0;
    std::string m_param1;
    std::vector<std::pair<double,double> > m_points;

  public:
    Contour();
    Contour(const std::string& name,
            const std::string& param0,
            const std::string& param1,
            const std::vector<std::pair<double,double> >& points);
    Contour(const GateInfo2D& band);
    Contour(const Contour& rhs);
    virtual ~Contour();

    GateType getType() const { return m_type; }
    std::unique_ptr<GateInfo> clone() const;

    void setParameter0(const std::string& paramName) { m_param0 = paramName; }
    std::string getParameter0() const { return m_param0; }

    void setParameter1(const std::string& paramName) { m_param1 = paramName; }
    std::string getParameter1() const { return m_param1; }

    std::vector<std::pair<double, double> > getPoints() const { return m_points; }
    std::vector<std::pair<double, double> >& getPoints() { return m_points; }
    std::pair<double, double> getPoint(size_t index) const
    { return m_points.at(index); }

    void setPoints(const std::vector<std::pair<double, double> >& points) {
        m_points = points;
    }
    void setPoint(size_t index, double x, double y);

    virtual bool operator==(const Contour& rhs) const;
    virtual bool operator!=(const Contour& rhs) const;
  }; // end of class

} // end of namespace

extern std::ostream& operator<<(std::ostream& stream,  const SpJs::Contour& cont);

#endif
